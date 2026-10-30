"""Evaluate a complete trained HQS checkpoint and a filter-off diagnostic."""
import common
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from infer_fixed_prior import metrics,sync
from train_native_hqs import Config,HQSUnfold,NativeData,physics_bank,seed_all,code_hashes,SCHEMA
from verify_filter_fade import grouped_mean,comparison

MODES=('trained_soft','trained_filter_off_diagnostic')
WITHOUT_PRIOR='trained_without_prior_diagnostic'


def predict(model,visibility,physics,mode):
    if mode not in (*MODES,WITHOUT_PRIOR): raise ValueError('Unknown evaluation mode')
    return model(visibility,physics,
                 projection='none' if mode=='trained_filter_off_diagnostic' else 'soft',
                 disable_prior=mode==WITHOUT_PRIOR,trace=True)


def load_full_model(checkpoint,device='cpu'):
    if checkpoint.get('schema')!=SCHEMA or checkpoint.get('phase')!='unfold':
        raise ValueError('Expected a full native128 unfold checkpoint')
    cfg=Config(**checkpoint['identity']['model_config']).validate()
    projection=checkpoint['identity']['training']['projection']
    if projection!='soft': raise ValueError('Expected the trained soft-filter configuration')
    model=HQSUnfold(cfg,'diffusion',projection).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,cfg


@torch.no_grad()
def run(args):
    seed_all(20261023)
    modes=tuple(args.modes)
    if not modes or len(set(modes))!=len(modes): raise ValueError('Choose unique modes')
    base=Path(args.baseline).resolve()
    previous=json.loads((base/'identity.json').read_text(encoding='utf-8'))
    prior=torch.load(base/'prior_frozen.pt',map_location='cpu',weights_only=False)
    source=Path(args.checkpoint).resolve()
    out=common.new_output(args.out,[base,source,previous['data']])
    before=common.sha(source)
    frozen=out/'unfold_frozen.pt';shutil.copyfile(source,frozen)
    if before!=common.sha(frozen) or before!=common.sha(source):
        raise ValueError('Checkpoint changed during snapshot')
    checkpoint=torch.load(frozen,map_location='cpu',weights_only=False)
    model,cfg=load_full_model(checkpoint,args.device)
    info=checkpoint['identity']
    if json.loads((source.parent/'identity.json').read_text(encoding='utf-8'))!=info:
        raise ValueError('Imported sidecar identity differs from checkpoint')
    if code_hashes()!=info['code']:
        raise ValueError('Imported training code identity differs from local implementation')
    data=NativeData(previous['data'],20)
    if data.identity!=info['data_core'] or data.selected_val!=previous['scenes']:
        raise ValueError('Dataset or validation selection differs')
    if info['model_config']!=previous['config']:
        raise ValueError('Baseline model configuration differs')
    if info.get('prior_sha256')!=common.sha(base/'prior_frozen.pt'):
        raise ValueError('Prior identity differs; rebuild same-prior controls first')
    if any(not torch.equal(value,checkpoint['model']['prior.'+name]) for name,value in prior['model'].items()):
        raise ValueError('Embedded prior tensors differ from original control')
    imported_environment=json.loads((source.parent/'environment.json').read_text(encoding='utf-8'))
    if imported_environment['val_scene_ids']!=data.selected_val:
        raise ValueError('Imported server validation scenes differ')
    validation=json.loads((source.parent/'latest_validation.json').read_text(encoding='utf-8'))
    if validation['step']!=checkpoint['step']:
        raise ValueError('Server validation sidecar is not from this best checkpoint')
    gammas=info['training']['gammas']
    bank=physics_bank(data,cfg,gammas,args.device,info['training'].get('cg_inner_tolerance_factor',.25))
    initial=HQSUnfold(cfg)
    learned={name:t.detach().cpu().tolist() for name,t in zip(('beta','mu','sigma','alpha'),model.stage_parameters())}
    fixed={name:t.detach().cpu().tolist() for name,t in zip(('beta','mu','sigma','alpha'),initial.stage_parameters())}
    identity=dict(method='full trained HQS checkpoint; paired filter-off mechanism diagnostic',
        checkpoint_sha256=before,checkpoint_step=checkpoint['step'],server_best_validation=checkpoint['best'],
        prior_sha256=info['prior_sha256'],same_prior_tensor_count=len(prior['model']),
        baseline=str(base),baseline_results_sha256=common.sha(base/'results.jsonl'),
        config=cfg.to_dict(),learned_parameters=learned,initial_parameters=fixed,
        training_identity=info,data=str(data.root),data_core=data.identity,scenes=data.selected_val,
        modes=modes,gammas=gammas,test_evaluated=False,
        diagnostic_note='Disabling prior removes all learned image correction; only data steps remain',
        device=args.device,torch=torch.__version__,cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
        imported_environment=imported_environment,server_validation=validation,
        seed=20261023,local_spectral_scales={str(g):float(p.spectral_scale) for g,p in bank.items()},
        inference_sha256=common.sha(__file__))
    common.write_json(out/'identity.json',identity)
    for name in ('identity.json','environment.json','latest_validation.json'):
        dest=out/'imported_metadata'/name;dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source.parent/name,dest)
    for name in [*info['code'],'infer_fixed_prior.py','verify_filter_fade.py',Path(__file__).name,'test_evaluate_unfold.py']:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(common.ROOT/name,dest)
    print(json.dumps(dict(step=checkpoint['step'],prior_equal=True,learned=learned)),flush=True)
    for p in bank.values():
        p.calls=dict(forward_calls=0,adjoint_calls=0)
        for method,key in [('forward','forward_calls'),('adjoint','adjoint_calls')]:
            original=getattr(p.operator,method)
            def counted(*a,_original=original,_key=key,_p=p,**kw):
                _p.calls[_key]+=1
                return _original(*a,**kw)
            setattr(p.operator,method,counted)
    rows=[];total=len(data.selected_val)*len(gammas)*len(modes)
    with (out/'results.jsonl').open('w',encoding='utf-8') as log:
        for scene in data.selected_val:
            for gamma,p in bank.items():
                truth,y,sigma=data.batch([scene],args.device,gamma)
                for mode in modes:
                    p.cg_records=[];p.calls=dict(forward_calls=0,adjoint_calls=0)
                    row=dict(scene=scene,gamma=gamma,mode=mode)
                    sync(args.device);start=time.perf_counter()
                    try:
                        pred_z,trace=predict(model,y,p,mode)
                        sync(args.device)
                        row.update(seconds=time.perf_counter()-start,**p.calls)
                        pred=cfg.denormalize(pred_z)
                        if not bool(torch.isfinite(pred).all()): raise RuntimeError('Non-finite prediction')
                        row.update(metrics(pred,truth,p.operator(pred)-y,sigma),status='ok',trace=trace,
                                   prior_calls=0 if mode==WITHOUT_PRIOR else cfg.stages,
                                   max_cg_relative_residual=max(p.cg_records))
                        path=out/'predictions'/mode/str(gamma)/(scene+'.npy')
                        path.parent.mkdir(parents=True,exist_ok=True)
                        np.save(path,pred.cpu().numpy()[0,0],allow_pickle=False)
                    except RuntimeError as exc:
                        row.update(status='failed',error=str(exc),seconds=time.perf_counter()-start,**p.calls)
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    common.write_json(out/'status.json',dict(completed=len(rows),total=total))
                    if len(rows)%40==0: print(f'{len(rows)}/{total}',flush=True)
    keys=('rmse_k','ssim','psnr_db','bias_k','standardized_residual_energy','seconds','forward_calls','adjoint_calls')
    result={}
    for mode in modes:
        good=[r for r in rows if r['mode']==mode and r['status']=='ok']
        result[mode]=dict(attempted=total//len(modes),failed=total//len(modes)-len(good),
            class_gamma_equal={k:grouped_mean(good,k) for k in keys},
            by_gamma={str(g):{k:grouped_mean([r for r in good if r['gamma']==g],k) for k in keys} for g in gammas})
    if all(r['status']=='ok' for r in rows) and 'trained_soft' in modes:
        oldrows=[json.loads(s) for s in (base/'results.jsonl').read_text().splitlines()]
        trained=[r for r in rows if r['mode']=='trained_soft']
        result['paired_comparisons']={mode:comparison(trained,[r for r in oldrows if r['mode']==mode])
                                      for mode in ('prior_unfiltered','prior_soft')}
        if MODES[1] in modes:
            result['filter_off_vs_trained_soft']=comparison(
                [r for r in rows if r['mode']==MODES[1]],trained)
        local=result['trained_soft']['class_gamma_equal']['rmse_k']
        result['local_minus_server_rmse_k']=local-validation['selection']
    common.write_json(out/'summary.json',result)
    common.write_json(out/'status.json',dict(status='complete',completed=len(rows),total=total,
                                            failed=sum(r['status']!='ok' for r in rows)))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--modes',nargs='+',choices=(*MODES,WITHOUT_PRIOR),default=list(MODES))
    run(parser.parse_args())
