"""Autograd forward/real-adjoint with the same float64 geometry and weights.

No custom backward; the tensor graph independently exposes gradient errors.
Supports leading batch dimensions, x is K. Float32 is an explicit option.
"""
import common
import numpy as np
import torch
from torch import nn

class TorchVisibility(nn.Module):
    def __init__(self, op, dtype=torch.float64, device='cpu'):
        super().__init__()
        cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
        self.n=op.n
        for name, arr, dt in [('w',op.w,dtype),('bx',op.b_xi,cdtype),('av',op.a_eta,cdtype),
                              ('iu',op.iu,torch.long),('iv',op.iv,torch.long)]:
            self.register_buffer(name,torch.as_tensor(np.array(arr,copy=True),dtype=dt,device=device))

    def forward(self,x):
        if x.shape[-2:]!=(self.n,self.n):
            raise ValueError('Wrong image grid')
        grid=self.bx @ (x*self.w).to(self.bx.dtype) @ self.av.T
        return grid[...,self.iu,self.iv]

    def adjoint(self,y):
        shape=(*y.shape[:-1],self.bx.shape[0]*self.av.shape[0])
        grid=torch.zeros(shape,dtype=self.bx.dtype,device=y.device)
        flat=self.iu*self.av.shape[0]+self.iv
        grid=grid.index_add(-1,flat,y)
        grid=grid.reshape(*y.shape[:-1],self.bx.shape[0],self.av.shape[0])
        return (self.bx.conj().T @ grid @ self.av.conj()).real*self.w

    def dc(self,x,y,step):
        return torch.clamp(x-step*self.adjoint(self(x)-y),150.,320.)
