"""Auto-select U-Net/Mamba from a prior or full HQS checkpoint; validation only."""
import common
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_native_hqs import NativeData,physics_bank,seed_all,code_hashes
from infer_fixed_prior import metrics,sync
from selectable_prior.checkpoints import load_model,checked_snapshot,source_hashes,SCHEMA


def grouped(rows,key):
    groups={}
    for row in rows:
        if row[key] is not None: groups.setdefault((row['scene'].split('/')[0],row.get('gamma')),[]).append(row[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


@torch.no_grad()
def main(a):
    if not 1<=a.val_per_class<=20: raise ValueError('val-per-class must be 1..20')
    seed_all(20261023)
    data=NativeData(a.data,a.val_per_class)
    out=common.new_output(a.out,[a.data,a.checkpoint])
    snapshot=out/'checkpoint_frozen.pt';digest=checked_snapshot(a.checkpoint,snapshot)
    state=torch.load(snapshot,map_location='cpu',weights_only=False)
    model,cfg,arch=load_model(state,data_core=data.identity,device=a.device)
    if (cfg.image_size,cfg.t_min,cfg.t_max)!=(128,150.,320.):
        raise ValueError('Evaluation requires native128 and 150..320 K')
    expected=source_hashes() if state['schema']==SCHEMA else code_hashes()
    if state['identity']['code']!=expected:
        raise ValueError('Checkpoint source differs; use its original source snapshot')
    model.eval().requires_grad_(False)
    phase=state['phase'];t=state['identity']['training']
    common.write_json(out/'identity.json',dict(architecture=arch,checkpoint_sha256=digest,
        checkpoint_step=state['step'],checkpoint_phase=phase,model_config=cfg.to_dict(),
        data_core=data.identity,scenes=data.selected_val,source_hashes=source_hashes(),
        torch=torch.__version__,cuda=torch.version.cuda,device=a.device,
        gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,test_evaluated=False))
    rows=[]
    if phase=='prior':
        generator=torch.Generator(device=a.device).manual_seed(20261023)
        for scene in data.selected_val:
            truth=data.batch([scene],a.device);clean=cfg.normalize(truth)
            noise=torch.randn(clean.shape,device=a.device,generator=generator)
            for sigma in (.02,.1,.25,.5):
                pred=cfg.denormalize(model(clean+sigma*noise,sigma))
                if not bool(torch.isfinite(pred).all()):raise RuntimeError('Nonfinite prior output')
                rows.append(dict(scene=scene,sigma=sigma,
                    before_rmse_k=float((noise*sigma*cfg.scale).square().mean().sqrt()),
                    rmse_k=float((pred-truth).square().mean().sqrt()),bias_k=float((pred-truth).mean())))
        summary={k:grouped(rows,k) for k in ('rmse_k','bias_k')}
        summary['by_sigma']={str(s):{k:grouped([r for r in rows if r['sigma']==s],k)
                                   for k in ('before_rmse_k','rmse_k','bias_k')} for s in (.02,.1,.25,.5)}
    else:
        bank=physics_bank(data,cfg,t['gammas'],a.device,t.get('cg_inner_tolerance_factor',.25))
        for scene in data.selected_val:
            for gamma,p in bank.items():
                truth,y,sigma=data.batch([scene],a.device,gamma);p.cg_records=[]
                sync(a.device)
                import time
                start=time.perf_counter();z,trace=model(y,p,trace=True);sync(a.device)
                seconds=time.perf_counter()-start;pred=cfg.denormalize(z)
                if not bool(torch.isfinite(pred).all()):raise RuntimeError('Nonfinite unfolded output')
                row=dict(scene=scene,gamma=gamma,**metrics(pred,truth,p.operator(pred)-y,sigma),
                         seconds=seconds,trace=trace,max_cg_relative_residual=max(p.cg_records))
                rows.append(row)
                dest=out/'predictions'/str(gamma)/(scene+'.npy');dest.parent.mkdir(parents=True,exist_ok=True)
                np.save(dest,pred.cpu().numpy()[0,0],allow_pickle=False)
                common.write_json(out/'status.json',dict(completed=len(rows),total=len(data.selected_val)*len(bank)))
        summary={k:grouped(rows,k) for k in ('rmse_k','ssim','psnr_db','bias_k','standardized_residual_energy','seconds')}
    with (out/'results.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    common.write_json(out/'summary.json',summary)
    common.write_json(out/'status.json',dict(status='complete',records=len(rows),test_evaluated=False))
    print(json.dumps(dict(architecture=arch,summary=summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--val-per-class',type=int,default=20)
    main(p.parse_args())
