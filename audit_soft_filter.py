"""Audit existing filtering without changing training or tuning parameters."""
import common
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from train_native_hqs import NativeData,Config,HQSUnfold,physics_bank,seed_all
from sair_unfold.numerics import cg


def relative(a,b):
    return float((a-b).norm()/b.norm().clamp_min(1e-20))


def precision_copy(physics):
    result=copy.deepcopy(physics)
    for module in result.modules():
        for name,value in list(module._buffers.items()):
            if value is not None and (value.is_complex() or value.is_floating_point()):
                module._buffers[name]=value.to(torch.complex128 if value.is_complex() else torch.float64)
    return result


@torch.no_grad()
def run(args):
    seed_all(20261023)
    base=Path(args.run)
    identity=json.loads((base/'identity.json').read_text(encoding='utf-8'))
    selection=json.loads((base/'visualizations/selection.json').read_text(encoding='utf-8'))
    cfg=Config(**identity['config']).validate()
    data=NativeData(identity['data'],20)
    ckpt=torch.load(base/'prior_frozen.pt',map_location='cpu',weights_only=False)
    if common.sha(base/'prior_frozen.pt')!=identity['prior_sha256']:
        raise ValueError('Frozen prior identity differs')
    model=HQSUnfold(cfg).to(args.device).eval().requires_grad_(False)
    model.prior.load_state_dict(ckpt['model'],strict=True)
    bank=physics_bank(data,cfg,identity['gammas'],args.device)
    invariants=[]; precision=[]; effects=[]
    beta,mu,sigmas,alpha=model.stage_parameters()
    for gamma,p in bank.items():
        u=torch.randn(1,1,128,128,device=args.device)
        v=torch.randn_like(u)
        qu,_=p.filter(u,mu[-1],'soft');qv,_=p.filter(v,mu[-1],'soft')
        left=(u*qv).sum();right=(qu*v).sum()
        symmetry=float((left-right).abs()/torch.maximum(u.norm()*qv.norm(),qu.norm()*v.norm()).clamp_min(1e-20))
        contraction=float(qu.norm()/u.norm())
        data_contraction=float(p(qu).norm()/p(u).norm())
        positive=float((u*qu).sum())
        if symmetry>1e-5 or contraction>1.00001 or data_contraction>1.00001 or positive<0:
            raise RuntimeError('Filter invariant failed')
        invariants.append(dict(gamma=gamma,symmetry_error=symmetry,
                               norm_ratio=contraction,measurement_norm_ratio=data_contraction,
                               positive_quadratic_form=positive,spectral_scale=float(p.spectral_scale)))
        for scene in selection['representatives']:
            truth,y,_=data.batch([scene],args.device,gamma)
            encoded=p.encode(y)
            zero=torch.zeros_like(truth)
            z,_=p.data_step(zero,encoded,zero.new_tensor(cfg.initial_beta))
            for k in range(cfg.stages):
                x,_=p.data_step(z,encoded,beta[k])
                d=model.prior(x,sigmas[k])-x
                q,info=p.filter(d,mu[k],'soft')
                # Compare both corrections at exactly the same pre-update x,d.
                unfiltered=cfg.denormalize(x+alpha[k]*d)
                filtered=cfg.denormalize(x+alpha[k]*q)
                effects.append(dict(scene=scene,gamma=gamma,stage=k,mu=float(mu[k]),
                    correction_norm_ratio=float(q.norm()/d.norm().clamp_min(1e-20)),
                    correction_data_ratio=float(p(q).norm()/p(d).norm().clamp_min(1e-20)),
                    unfiltered_rmse_k=float((unfiltered-truth).square().mean().sqrt()),
                    filtered_rmse_k=float((filtered-truth).square().mean().sqrt()),
                    removed_truth_alignment=float(((d-q)*(cfg.normalize(truth)-x)).sum()),
                    cg_relative_residual=float(info['relative_residual'].max())))
                if scene==selection['representatives'][0] and gamma in (1.,.35) and k in (0,7):
                    p64=precision_copy(p)
                    d64=d.double();m64=mu[k].double()
                    reference,reference_info=cg(lambda w:p64.normal(w)+m64*w,m64*d64,
                                                iterations=512,rtol=1e-11)
                    error=relative(q.double(),reference)
                    precision.append(dict(scene=scene,gamma=gamma,stage=k,
                        filter_relative_error_vs_float64=error,
                        update_rmse_error_k=float(((q.double()-reference)*cfg.scale*alpha[k]).square().mean().sqrt()),
                        reference_residual=float(reference_info['relative_residual'].max())))
                    if error>2e-4 or float(reference_info['relative_residual'].max())>1e-10:
                        raise RuntimeError('Precision reference failed')
                z=x+alpha[k]*q
    out=common.new_output(args.out,[base/'prior_frozen.pt',identity['data']])
    common.write_json(out/'audit.json',dict(prior_sha256=identity['prior_sha256'],device=args.device,
        invariant_checks=invariants,precision_checks=precision,same_state_comparisons=effects,
        source_hashes={str(p.relative_to(common.ROOT)):common.sha(p) for p in
                      [common.ROOT/'vendor/sair_unfold/physics.py',common.ROOT/'torch_operator.py',Path(__file__)]}))
    summary=dict(invariants_passed=len(invariants),float64_checks_passed=len(precision),
        max_float64_filter_relative_error=max(r['filter_relative_error_vs_float64'] for r in precision),
        max_update_rmse_error_k=max(r['update_rmse_error_k'] for r in precision),
        same_state_comparisons=len(effects),
        filtering_improves_rmse_count=sum(r['filtered_rmse_k']<r['unfiltered_rmse_k'] for r in effects),
        mean_rmse_delta_k=float(np.mean([r['filtered_rmse_k']-r['unfiltered_rmse_k'] for r in effects])),
        mean_correction_norm_ratio=float(np.mean([r['correction_norm_ratio'] for r in effects])),
        mean_measurement_change_ratio=float(np.mean([r['correction_data_ratio'] for r in effects])))
    common.write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    run(parser.parse_args())
