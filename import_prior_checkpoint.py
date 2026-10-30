"""Explicitly relocate our own native128 prior checkpoints to a server.

Checks code bytes, configuration and immutable data hashes before rebinding
only platform-dependent root paths / path separators. No tensor is changed.
"""
import common
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_native_hqs import NativeData,code_hashes,atomic_save
from sair_unfold.config import Config

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def canonical(code):return {k.replace('\\','/'):v for k,v in code.items()}

def main(a):
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    data=NativeData(cfg['data'],cfg['training']['val_scenes_per_class'])
    source=Path(a.checkpoints).resolve()
    states={name:torch.load(source/name,map_location='cpu',weights_only=False) for name in ('best.pt','last.pt')}
    newcode=code_hashes()
    if states['best.pt']['step']>states['last.pt']['step']:raise ValueError('Checkpoint pair was copied during an update; copy last.pt again')
    for state in states.values():
        ident=state['identity']
        if state.get('schema')!='native128-hqs-training-v1' or state.get('phase')!='prior':raise ValueError('Expected native128 prior checkpoints')
        if ident['model_config']!=Config(**cfg['model']).to_dict() or ident['training']!=cfg['training']:raise ValueError('Training config differs')
        if ident['data_core']!=data.identity:raise ValueError('Data bytes or metadata changed')
        if canonical(ident['code'])!=canonical(newcode):raise ValueError('Training source bytes differ; use the matching server bundle')
    out=common.new_output(a.out,[source,data.root]);(out/'prior').mkdir()
    common.write_json(out/'run_config.json',cfg)
    records=[]
    for name,state in states.items():
        old_root=state['identity']['data_root']
        state['identity']['data_root']=str(data.root);state['identity']['code']=newcode
        atomic_save(out/'prior'/name,state)
        reread=torch.load(out/'prior'/name,map_location='cpu',weights_only=False)
        for key in ('model','optimizer','scaler','rng','sample_position','step','best'):
            if not same(state[key],reread[key]):raise AssertionError('Checkpoint content changed: '+key)
        records.append(dict(file=name,step=state['step'],old_data_root=old_root,new_data_root=str(data.root),
                            exact_tensor_optimizer_rng_transfer=True))
    common.write_json(out/'server_import.json',dict(status='pass',checkpoints=records,
        limitation='Cross OS/GPU/library runs need not be bitwise identical; new server environment is recorded by training'))
    print(json.dumps(dict(status='pass',resume_step=states['last.pt']['step'],run=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/server_128.json');p.add_argument('--checkpoints',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
