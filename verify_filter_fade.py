"""One prespecified fading-filter validation; frozen prior, no training."""
import common
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from infer_fixed_prior import metrics, sync
from train_native_hqs import NativeData,Config,HQSUnfold,physics_bank,seed_all,code_hashes

WEIGHTS=(1.,1.,1.,1.,.75,.5,.25,0.)


def mixed_correction(d,q,weight):
    if not 0 <= weight <= 1: raise ValueError('Filter mixture must be in [0,1]')
    if weight==0: return d
    if weight==1: return q
    return weight*q+(1-weight)*d


@torch.no_grad()
def reconstruct(model,p,visibility,weights=WEIGHTS,trace=True):
    if len(weights)!=model.config.stages or any(not 0<=w<=1 for w in weights):
        raise ValueError('Expected one [0,1] filter weight per stage')
    encoded=p.encode(visibility)
    zero=visibility.real.new_zeros(len(visibility),1,model.config.image_size,model.config.image_size)
    z,_=p.data_step(zero,encoded,zero.new_tensor(model.config.initial_beta))
    beta,mu,sigma,alpha=model.stage_parameters()
    records=[]
    for k,weight in enumerate(weights):
        x,dc=p.data_step(z,encoded,beta[k])
        d=model.prior(x,sigma[k])-x
        if weight:
            q,info=p.filter(d,mu[k],'soft')
        else:
            q,info=d,{'relative_residual':torch.zeros(1,device=d.device)}
        correction=mixed_correction(d,q,weight)
        z=x+alpha[k]*correction
        if trace:
            before=p.operator(model.config.denormalize(x))-visibility
            after=p.operator(model.config.denormalize(z))-visibility
            records.append(dict(stage=k,filter_weight=weight,beta=float(beta[k]),mu=float(mu[k]),
                sigma=float(sigma[k]),alpha=float(alpha[k]),
                dc_cg_residual=float(dc['relative_residual'].max()),
                filter_cg_residual=float(info['relative_residual'].max()),
                visibility_mse_before_prior=float(before.abs().square().mean()),
                visibility_mse_after_prior=float(after.abs().square().mean()),
                prior_visibility_change_rms=float(p.operator(correction).abs().square().mean().sqrt()*model.config.scale*alpha[k])))
    return z,records


def grouped_mean(rows,key):
    groups={}
    for row in rows:
        if row.get(key) is not None:
            groups.setdefault((row['scene'].split('/')[0],row['gamma']),[]).append(row[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def comparison(new,baseline):
    matched={(r['scene'],r['gamma']):r for r in baseline}
    result={}
    for key in ('rmse_k','ssim','standardized_residual_energy'):
        pairs={}
        for r in new:
            delta=r[key]-matched[(r['scene'],r['gamma'])][key]
            pairs.setdefault(r['scene'],[]).append(delta)
        scene_delta={scene:float(np.mean(values)) for scene,values in pairs.items()}
        groups={}
        for scene,delta in scene_delta.items(): groups.setdefault(scene.split('/')[0],[]).append(delta)
        # Resample scenes within each fixed class, keeping its gamma results together.
        rng=np.random.default_rng(20261025)
        estimates=np.mean([rng.choice(v,size=(2000,len(v)),replace=True).mean(axis=1)
                           for v in groups.values()],axis=0)
        deltas=[r[key]-matched[(r['scene'],r['gamma'])][key] for r in new]
        result[key]=dict(mean_delta=float(np.mean([np.mean(v) for v in groups.values()])),
            improved_pairs=sum(x>0 if key=='ssim' else x<0 for x in deltas),pairs=len(deltas),
            scene_stratified_bootstrap_95_interval=np.quantile(estimates,[.025,.975]).tolist())
    return result


@torch.no_grad()
def run(args):
    seed_all(20261023)
    base=Path(args.baseline).resolve()
    identity=json.loads((base/'identity.json').read_text(encoding='utf-8'))
    status=json.loads((base/'status.json').read_text(encoding='utf-8'))
    if status.get('status')!='complete' or status.get('failed')!=0:
        raise ValueError('Expected a complete baseline')
    if args.device!=identity['device']: raise ValueError('Use the original device for paired validation')
    current=code_hashes()
    if current!=identity['source_hashes'] or common.sha(common.ROOT/'infer_fixed_prior.py')!=identity['inference_sha256']:
        raise ValueError('Baseline code changed; do not silently compare')
    if torch.__version__!=identity['torch']: raise ValueError('PyTorch environment differs')
    prior=base/'prior_frozen.pt'
    if common.sha(prior)!=identity['prior_sha256']: raise ValueError('Prior identity differs')
    cfg=Config(**identity['config']).validate()
    if cfg.stages!=8: raise ValueError('Eight stages required')
    data=NativeData(identity['data'],20)
    if data.identity!=identity['data_core'] or data.selected_val!=identity['scenes']:
        raise ValueError('Validation identity differs')
    out=common.new_output(args.out,[base,identity['data']])
    ckpt=torch.load(prior,map_location='cpu',weights_only=False)
    model=HQSUnfold(cfg).to(args.device).eval().requires_grad_(False)
    model.prior.load_state_dict(ckpt['model'],strict=True)
    bank=physics_bank(data,cfg,identity['gammas'],args.device)
    run_identity=dict(identity,method='fixed VE/HQS, prespecified fading filter',
        baseline=str(base),baseline_results_sha256=common.sha(base/'results.jsonl'),
        filter_weights=list(WEIGHTS),prespecified_candidate_count=1,
        prior_file=str(prior),prior_updates_performed=0,
        purpose='validation follow-up informed by previous validation audit; not independent test',
        stage_formula='d_effective = lambda_k * Q_mu(d) + (1-lambda_k) * d',
        spectral_scale_by_gamma={str(g):float(p.spectral_scale) for g,p in bank.items()},
        experiment_sha256=common.sha(__file__))
    common.write_json(out/'identity.json',run_identity)
    source=out/'source';source.mkdir()
    for name in [*current,'infer_fixed_prior.py',Path(__file__).name,'test_filter_fade.py']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(common.ROOT/name,target)
    baseline_rows=[json.loads(s) for s in (base/'results.jsonl').read_text().splitlines()]
    anchors=[]
    scene=data.selected_val[0]
    for gamma,p in bank.items():
        truth,y,_=data.batch([scene],args.device,gamma)
        for mode,weight in [('prior_unfiltered',0.),('prior_soft',1.)]:
            original=model(y,p,projection='none' if weight==0 else 'soft',trace=False)
            replicated,_=reconstruct(model,p,y,(weight,)*8,trace=False)
            torch.testing.assert_close(replicated,original,rtol=0,atol=0)
            recorded=torch.from_numpy(np.load(base/'predictions'/mode/str(gamma)/(scene+'.npy'))).to(args.device)
            difference=float((cfg.denormalize(replicated)[0,0]-recorded).abs().max())
            if difference>1e-4: raise RuntimeError(f'Baseline anchor differs by {difference} K')
            anchors.append(dict(scene=scene,gamma=gamma,mode=mode,max_saved_prediction_difference_k=difference))
    common.write_json(out/'anchors.json',anchors)
    print('All 8 endpoint and saved-baseline anchors passed',flush=True)
    for p in bank.values():
        p.calls=dict(forward_calls=0,adjoint_calls=0)
        for method,key in [('forward','forward_calls'),('adjoint','adjoint_calls')]:
            original=getattr(p.operator,method)
            def counted(*a,_original=original,_key=key,_p=p,**kw):
                _p.calls[_key]+=1
                return _original(*a,**kw)
            setattr(p.operator,method,counted)
    rows=[];total=len(data.selected_val)*len(bank)
    with (out/'results.jsonl').open('w',encoding='utf-8') as log:
        for scene in data.selected_val:
            for gamma,p in bank.items():
                truth,y,sigma=data.batch([scene],args.device,gamma)
                p.calls=dict(forward_calls=0,adjoint_calls=0);p.cg_records=[]
                row=dict(scene=scene,gamma=gamma,mode='prior_fade')
                sync(args.device);start=time.perf_counter()
                try:
                    z,trace=reconstruct(model,p,y)
                    sync(args.device)
                    row.update(seconds=time.perf_counter()-start,**p.calls)
                    pred=cfg.denormalize(z)
                    if not bool(torch.isfinite(pred).all()): raise RuntimeError('Non-finite prediction')
                    row.update(metrics(pred,truth,p.operator(pred)-y,sigma),status='ok',trace=trace,
                               prior_calls=8,max_cg_relative_residual=max(p.cg_records))
                    dest=out/'predictions/prior_fade'/str(gamma)/(scene+'.npy')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    np.save(dest,pred.cpu().numpy()[0,0],allow_pickle=False)
                except RuntimeError as exc:
                    row.update(status='failed',error=str(exc),seconds=time.perf_counter()-start,**p.calls)
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                common.write_json(out/'status.json',dict(completed=len(rows),total=total))
                if len(rows)%20==0: print(f'{len(rows)}/{total}',flush=True)
    good=[r for r in rows if r['status']=='ok']
    keys=('rmse_k','ssim','psnr_db','bias_k','standardized_residual_energy','seconds','forward_calls','adjoint_calls')
    summary=dict(attempted=total,failed=total-len(good),class_gamma_equal={k:grouped_mean(good,k) for k in keys},
                 by_gamma={str(g):{k:grouped_mean([r for r in good if r['gamma']==g],k) for k in keys} for g in bank})
    if len(good)==total:
        summary['paired_comparisons']={mode:comparison(good,[r for r in baseline_rows if r['mode']==mode])
                                      for mode in ('prior_unfiltered','prior_soft')}
    common.write_json(out/'summary.json',summary)
    common.write_json(out/'status.json',dict(status='complete',completed=len(rows),total=total,failed=total-len(good)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
