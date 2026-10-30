"""Check standalone imports, transfer hashes, GT/y pairing and the forward graph."""
import common
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from train_native_hqs import NativeData
from sair_experiments.operators import dataset_operator
from torch_operator import TorchVisibility

def main(a):
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    expected=json.loads((common.ROOT/'dataset_transfer_manifest.json').read_text(encoding='utf-8'))
    root=Path(cfg['data']).resolve()
    for name,digest in expected['required_files'].items():
        if common.sha(root/name)!=digest:raise ValueError('Transferred metadata changed: '+name)
    counts={name:sum(1 for _ in (root/name).rglob('*.npy' if name=='gt' else '*.npz')) for name in ('gt','meas')}
    if counts!={'gt':31500,'meas':31500}:raise ValueError('Missing GT/measurement files: '+str(counts))
    data=NativeData(root,cfg['training']['val_scenes_per_class'])
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU PyTorch/CUDA unavailable')
    maxima=[]
    for scene in (data.train[0],data.val[0]):
        for gamma in cfg['training']['gammas']:
            op=dataset_operator(data.reader,gamma)
            truth=data.truth(scene)[0].astype(float)
            model=TorchVisibility(op,device=a.device)
            tx=torch.tensor(truth,device=a.device,dtype=torch.float64,requires_grad=True)
            prediction=model(tx)
            reference=op(truth)
            error=float(np.max(abs(prediction.detach().cpu().numpy()-reference)))
            if error>1e-9:raise ValueError('Tensor and reference forward differ')
            loss=.5*prediction.abs().square().sum()
            derivative=torch.autograd.grad(loss,tx)[0].detach().cpu().numpy()
            if np.max(abs(derivative-op.adjoint(reference)))>1e-9:raise ValueError('Adjoint/autograd mismatch')
            obs=data.reader.load(scene,gamma)
            residual=float(np.sqrt(np.mean(abs(reference-obs['y'])**2))/obs['sigma'])
            if not .8<residual<1.2:raise ValueError('GT/y noise-scale pairing is implausible: '+scene)
            maxima.append(error)
    fingerprint=None
    if a.full_data_fingerprint:
        check=hashlib.sha256()
        scenes=data.reader.metadata['selected_scenes']
        for i,scene in enumerate(scenes,1):
            digest=common.sha(root/'gt'/(scene+'.npy'))+common.sha(root/'meas'/(scene+'.npz'))
            check.update((scene+'\n'+digest+'\n').encode())
            if i%5000==0:print(f'Content fingerprint {i}/{len(scenes)}',flush=True)
        fingerprint=check.hexdigest()
        if fingerprint!=expected['reference_content_fingerprint']:raise ValueError('Full data content fingerprint differs')
    print(json.dumps(dict(status='pass',python=sys.version,torch=torch.__version__,numpy=np.__version__,
        device=a.device,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
        data=str(root),files=counts,train_scenes=len(data.train),val_scenes=len(data.val),
        max_tensor_reference_error_k=max(maxima),test_reconstruction_evaluated=False,
        full_content_fingerprint=fingerprint),ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/server_128.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--full-data-fingerprint',action='store_true');main(p.parse_args())
