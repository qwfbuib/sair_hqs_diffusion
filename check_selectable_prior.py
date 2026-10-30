"""Finite real-data compatibility and resource checks; not a quality experiment."""
import common
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from train_native_hqs import (Config,NativeData,physics_bank,seed_all,code_hashes,
                              reconstruction_loss,optimizer_update)
from sair_unfold.prior import DiffusionPrior
from sair_unfold.unfolding import HQSUnfold
from selectable_prior.models import SelectablePrior,SelectableHQS,architecture
from selectable_prior.checkpoints import load_model


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def check(a):
    seed_all(20261023)
    out=common.new_output(a.out,[a.data])
    config=json.loads((common.ROOT/'configs/server_128_unet.json').read_text())
    cfg=Config(**config['model']).validate();data=NativeData(a.data,1)
    reference=json.loads((common.ROOT/'runs/unfold_val_step7000_v1/identity.json').read_text())
    if code_hashes()!=reference['training_identity']['code']:
        raise ValueError('Legacy source changed')
    compatibility=[]
    prior_path=common.ROOT/'runs/fixed_prior_val_v1/prior_frozen.pt'
    full_path=common.ROOT/'runs/unfold_val_step7000_v1/unfold_frozen.pt'
    state=torch.load(prior_path,map_location='cpu',weights_only=False)
    legacy=DiffusionPrior(cfg).to(a.device).eval();legacy.load_state_dict(state['model'])
    selected,_,_=load_model(state,expected_backbone='unet',data_core=data.identity,device=a.device)
    selected.eval();target=cfg.normalize(data.batch([data.selected_val[0]],a.device))
    with torch.no_grad():
        for sigma in (.02,.1,.25,.5):
            old=legacy(target,sigma);new=selected(target,sigma)
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            compatibility.append(dict(phase='prior',sigma=sigma,max_error=0.))
    del legacy,selected,state
    state=torch.load(full_path,map_location='cpu',weights_only=False)
    legacy=HQSUnfold(cfg).to(a.device).eval();legacy.load_state_dict(state['model'])
    selected,_,_=load_model(state,expected_backbone='unet',data_core=data.identity,device=a.device)
    selected.eval();bank=physics_bank(data,cfg,[1.,.75,.5,.35],a.device)
    with torch.no_grad():
        for gamma,p in bank.items():
            _,y,_=data.batch([data.selected_val[0]],a.device,gamma)
            old=legacy(y,p);new=selected(y,p)
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            compatibility.append(dict(phase='unfold',gamma=gamma,max_error=0.,
                                      max_cg_residual=max(p.cg_records)))
    del legacy,selected,state,bank,p,old,new
    gc.collect()
    if a.device=='cuda':torch.cuda.empty_cache()
    resource=[]
    for backbone in ('unet','mamba'):
        seed_all(cfg.seed)
        if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
        prior=SelectablePrior(cfg,backbone).to(a.device).train()
        optimizer=torch.optim.AdamW(prior.parameters(),lr=cfg.learning_rate,weight_decay=0.)
        scaler=torch.cuda.amp.GradScaler(enabled=a.device=='cuda')
        generator=torch.Generator(device=a.device).manual_seed(20261023)
        target=cfg.normalize(data.batch(data.train[:a.batch],a.device))
        times=[];norms=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);sync(a.device);start=time.perf_counter()
            noise_state=generator.get_state()
            for attempt in range(16):
                generator.set_state(noise_state);optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=a.device,enabled=a.device=='cuda'):
                    loss=prior.dsm_loss(target,generator)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite prior loss')
                scaler.scale(loss).backward()
                norm,skipped=optimizer_update(optimizer,scaler,list(prior.parameters()),1.,a.device=='cuda')
                if not skipped:break
            if skipped:raise RuntimeError('Resource pilot AMP repeatedly overflowed')
            sync(a.device);times.append(time.perf_counter()-start);norms.append(norm)
        count=sum(p.numel() for p in prior.parameters())
        resource.append(dict(backbone=backbone,phase='prior',batch=a.batch,updates=2,
            parameters=count,seconds=times,gradient_norms=norms,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else None))
        print(json.dumps(resource[-1]),flush=True)
        model=SelectableHQS(cfg,backbone).to(a.device)
        model.prior.load_state_dict(prior.state_dict())
        del prior,optimizer,scaler,target,loss
        model.prior.requires_grad_(False);model.train();model.prior.eval()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=cfg.learning_rate,weight_decay=0.)
        bank=physics_bank(data,cfg,[1.],a.device);p=bank[1.]
        gc.collect()
        if a.device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True);sync(a.device);start=time.perf_counter()
        for scene in data.train[:4]:
            truth,y,_=data.batch([scene],a.device,1.)
            predicted,_=model(y,p,trace=True)
            loss,_=reconstruction_loss(predicted,cfg.normalize(truth),y,p,cfg)
            (loss/4).backward()
        if not all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in params):
            raise RuntimeError('Missing/nonfinite frozen-prior stage gradient')
        nonzero={name:float(q.grad.abs().sum()) for name,q in model.named_parameters() if q.requires_grad}
        if not all(v>0 for v in nonzero.values()):raise RuntimeError('Zero stage gradient')
        optimizer.step();sync(a.device)
        resource.append(dict(backbone=backbone,phase='unfold',batch=1,accumulation=4,updates=1,
            parameters=sum(q.numel() for q in model.parameters()),trainable_parameters=sum(q.numel() for q in params),
            seconds=time.perf_counter()-start,stage_gradient_l1=nonzero,max_cg_residual=max(p.cg_records),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else None))
        print(json.dumps(resource[-1]),flush=True)
        del model,optimizer,bank,p,params,predicted,loss,truth,y
        gc.collect()
        if a.device=='cuda':torch.cuda.empty_cache()
    result=dict(status='pass',architecture={b:architecture(b) for b in ('unet','mamba')},
        compatibility=compatibility,resource=resource,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,data_core=data.identity,
        original_source_hashes=code_hashes(),test_evaluated=False,
        limitation='Two prior updates and one unfolded update are engineering checks, not quality evidence')
    common.write_json(out/'report.json',result)
    print('Compatibility and full128 resource checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch',type=int,default=4)
    check(p.parse_args())
