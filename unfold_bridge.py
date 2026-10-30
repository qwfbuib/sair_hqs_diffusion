"""Reuse existing HQS modules with the audited native128 Y-array operator.

Replaces CrossSAIROperator; no legacy train/simulate loader is reused.
"""
import common
import torch
from torch_operator import TorchVisibility

class Native128YOperator(TorchVisibility):
    def __init__(self,op,device='cuda'):
        if op.n!=128:raise ValueError('Native128 required')
        super().__init__(op,dtype=torch.float32,device=device)
        self.image_size=128;self.measurements=len(op.u);self.identity=op.identity

    def forward(self,temperature):
        if temperature.ndim!=4 or temperature.shape[1:]!=(1,128,128):
            raise ValueError('Expected [B,1,128,128] K')
        return super().forward(temperature[:,0])

    def adjoint(self,y):
        return super().adjoint(y)[:,None]

    def metadata(self):
        return dict(kind='native128_y_array',n_grid=128,theta=1.05,
                    float64_geometry_identity=self.identity,measurements=self.measurements,
                    temperature_units='K',complex_noise='E|n|²=sigma²')
