"""Separate orchestration reusing legacy data, physics, losses and validation."""
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
import common
from train_native_hqs import (Config,NativeData,Cursor,physics_bank,validate,
    seed_all,rng_state,restore_rng,optimizer_update,atomic_save,atomic_json,
    reconstruction_loss)
from .models import SelectablePrior,SelectableHQS,architecture
from .checkpoints import SCHEMA,source_hashes,load_model,checked_snapshot


def backbone_from(config):
    prior=config.get('prior',{'backbone':'unet'})
    if not isinstance(prior,dict) or set(prior)!={'backbone'}:
        raise ValueError('prior must contain only backbone: unet or mamba')
    architecture(prior['backbone'])
    return prior['backbone']


def restore_training(state,identity,model,optimizer,scaler,noise):
    if state.get('schema')!=SCHEMA or state['identity']!=identity:
        raise ValueError('Resume identity differs; legacy or cross-backbone resume is forbidden')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng']);noise.set_state(state['noise_rng'].cpu())


def train_phase(phase,args,config,data):
    cfg=Config(**config['model']).validate();t=config['training']
    backbone=backbone_from(config);seed=cfg.seed+(100 if phase=='unfold' else 0)
    seed_all(seed);device=torch.device(args.device)
    output=Path(args.out).resolve();dest=output/phase
    if dest.exists() and not args.resume: raise FileExistsError('Use a new phase directory or --resume')
    if args.resume and dest.exists() and not (dest/'last.pt').exists():
        raise ValueError('No new-format last.pt exists for resume')
    dest.mkdir(parents=True,exist_ok=True)
    model=(SelectablePrior(cfg,backbone) if phase=='prior' else
           SelectableHQS(cfg,backbone,t['projection'])).to(device)
    prior_hash=None
    if phase=='unfold':
        source=Path(args.prior) if args.prior else output/'prior/best.pt'
        frozen=dest/'input_prior.pt'
        # Resume is bound to the original copied prior, even if external best changes.
        if args.resume and frozen.exists():
            prior_hash=common.sha(frozen)
            if args.prior and common.sha(source)!=prior_hash:
                raise ValueError('Explicit prior differs from resume input')
        else:
            if source.resolve().is_relative_to(dest): raise ValueError('Prior input overlaps this phase')
            prior_hash=checked_snapshot(source,frozen)
        state=torch.load(frozen,map_location='cpu',weights_only=False)
        loaded,prior_cfg,_=load_model(state,expected_backbone=backbone,
                                    expected_phase='prior',data_core=data.identity)
        if prior_cfg.to_dict()!=cfg.to_dict(): raise ValueError('Prior model configuration differs')
        model.prior.load_state_dict(loaded.state_dict(),strict=True)
        del loaded,state
        if t['freeze_prior']: model.prior.requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg.learning_rate,weight_decay=0.)
    use_amp=phase=='prior' and t['prior_amp'] and device.type=='cuda'
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    noise=torch.Generator(device=device).manual_seed(cfg.seed+314159)
    identity=dict(model_config=cfg.to_dict(),training=t,data_root=str(data.root),
                  data_core=data.identity,phase=phase,architecture=architecture(backbone),
                  prior_sha256=prior_hash,code=source_hashes(),noise_seed=cfg.seed+314159,
                  noise_policy='private generator; AMP retries replay same samples and noise')
    step=position=0;best=math.inf
    if args.resume and (dest/'last.pt').exists():
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        restore_training(state,identity,model,optimizer,scaler,noise)
        step,position,best=state['step'],state['sample_position'],state['best']
        del state
    cursor=Cursor(len(data.train)*(len(t['gammas']) if phase=='unfold' else 1),seed,position)
    bank=physics_bank(data,cfg,t['gammas'],device,t.get('cg_inner_tolerance_factor',.25)) if phase=='unfold' else None
    atomic_json(dest/'identity.json',identity)
    atomic_json(dest/'environment.json',dict(python=sys.version,torch=torch.__version__,
        executable=sys.executable,cuda=torch.version.cuda,device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),architecture=identity['architecture'],
        effective_batch=t[phase+'_batch_size']*(t['unfold_accumulation'] if phase=='unfold' else 1),
        amp=use_amp,train_scenes=len(data.train),val_scene_ids=data.selected_val,test_access=False))
    start=time.perf_counter()

    def save(name):
        atomic_save(dest/name,dict(schema=SCHEMA,phase=phase,identity=identity,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            step=step,sample_position=cursor.position,best=best,rng=rng_state(),noise_rng=noise.get_state()))

    total=t[phase+'_updates']
    if step>=total:return True
    retries=0
    with (dest/'metrics.jsonl').open('a',encoding='utf-8') as log:
        while step<total:
            before=time.perf_counter();model.train()
            if phase=='unfold' and t['freeze_prior']:model.prior.eval()
            replay=dict(position=cursor.position,rng=rng_state(),noise=noise.get_state())
            optimizer.zero_grad(set_to_none=True)
            acc=t['unfold_accumulation'] if phase=='unfold' else 1
            loss_sum=0.;gamma_values=[];worst=0.
            for _ in range(acc):
                batch=t[phase+'_batch_size'];chosen=cursor.take(batch)
                if phase=='prior':
                    target=cfg.normalize(data.batch([data.train[i] for i in chosen],device))
                    with torch.autocast(device_type=device.type,enabled=use_amp):
                        loss=model.dsm_loss(target,generator=noise)
                else:
                    if batch!=1:raise ValueError('Variable gamma observations require unfold batch1')
                    si,gi=divmod(chosen[0],len(t['gammas']));gamma=t['gammas'][gi]
                    target,y,_=data.batch([data.train[si]],device,gamma)
                    bank[gamma].cg_records=[]
                    predicted,_=model(y,bank[gamma],trace=True)
                    loss,_=reconstruction_loss(predicted,cfg.normalize(target),y,bank[gamma],cfg)
                    gamma_values.append(gamma);worst=max(worst,max(bank[gamma].cg_records))
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
                loss_sum+=float(loss.detach())/acc
                scaler.scale(loss/acc).backward()
            norm,skipped=optimizer_update(optimizer,scaler,parameters,t['gradient_clip'],use_amp)
            if skipped:
                cursor.position=replay['position'];restore_rng(replay['rng']);noise.set_state(replay['noise'])
                retries+=1
                log.write(json.dumps(dict(event='amp_retry_same_input',step=step,retry=retries))+'\n');log.flush()
                if retries>=16:raise RuntimeError('16 consecutive AMP overflow retries')
                continue
            retries=0;step+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-before;val=None
            if step==1 or step%t[phase+'_validate_every']==0 or step==total:
                val=validate(model,phase,data,cfg,t,device,bank)
                if val['selection']<best:best=val['selection'];save('best.pt')
                atomic_json(dest/'latest_validation.json',dict(step=step,**val))
            if (step==1 or step%t['checkpoint_every']==0 or val is not None
                    or (args.stop_after and step>=args.stop_after)):
                save('last.pt')
            row=dict(step=step,loss=loss_sum,gradient_norm=norm,step_seconds=seconds,
                     sample_position=cursor.position,validation=val,gammas=gamma_values,
                     max_cg_relative_residual=worst,backbone=backbone,
                     peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            atomic_json(output/'status.json',dict(status='running',phase=phase,step=step,total=total,
                backbone=backbone,elapsed=time.perf_counter()-start),required=False)
            if step==1 or step%t['log_every']==0 or val:
                print(f'{backbone}/{phase}: {step}/{total}, loss={loss_sum:.6f}, seconds={seconds:.3f}, val={val}',flush=True)
            if args.stop_after and step>=args.stop_after:
                return step>=total
    save('last.pt')
    return True


def main(args):
    import traceback
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.data:config['data']=args.data
    config['data']=str(Path(config['data']).resolve())
    backbone_from(config);cfg=Config(**config['model']).validate()
    if (cfg.image_size,cfg.t_min,cfg.t_max)!=(128,150.,320.):
        raise ValueError('This entrypoint requires native128 and 150..320 K')
    if args.stop_after is not None and args.stop_after<1:
        raise ValueError('stop-after must be a positive optimizer update count')
    t=config['training']
    if t['test_evaluation']:raise ValueError('Training does not evaluate test')
    if args.smoke:
        t.update(prior_updates=2,unfold_updates=2,prior_batch_size=1,unfold_batch_size=1,
                 unfold_accumulation=1,prior_validate_every=1,unfold_validate_every=1,
                 checkpoint_every=1,log_every=1,val_scenes_per_class=1,engineering_smoke=True)
    if any(not isinstance(t[k],int) or isinstance(t[k],bool) or t[k]<1 for k in
           ('prior_updates','unfold_updates','prior_batch_size','unfold_batch_size',
            'unfold_accumulation','prior_validate_every','unfold_validate_every',
            'checkpoint_every','log_every','val_scenes_per_class')):
        raise ValueError('Training counts and batch sizes must be positive integers')
    if t['unfold_batch_size']!=1 or not 1<=t['val_scenes_per_class']<=20:
        raise ValueError('Use unfold batch1 and 1..20 validation scenes per class')
    data=NativeData(config['data'],t['val_scenes_per_class'])
    out=Path(args.out).resolve()
    if out==data.root or out.is_relative_to(data.root) or data.root.is_relative_to(out):
        raise ValueError('Output overlaps dataset')
    resume_phase='prior' if args.phase=='pipeline' else args.phase
    if args.resume and not (out/resume_phase/'last.pt').is_file():
        raise ValueError(f'Resume requires an existing {resume_phase}/last.pt')
    if out.exists() and not (out/'run_config.json').exists():
        raise FileExistsError('Use a new output root')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_config.json').exists() and json.loads((out/'run_config.json').read_text())!=config:
        raise ValueError('Run configuration differs')
    atomic_json(out/'run_config.json',config)
    snapshot=out/'source'
    if not snapshot.exists():
        import shutil
        for name,h in source_hashes().items():
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(common.ROOT/name,dest)
            if common.sha(dest)!=h:raise ValueError('Source changed during snapshot')
    phases=('prior','unfold') if args.phase=='pipeline' else (args.phase,)
    try:
        for phase in phases:
            if not train_phase(phase,args,config,data):
                atomic_json(out/'status.json',dict(status='controlled_stop',phase=phase,test_evaluated=False));return
        atomic_json(out/'status.json',dict(status='complete',phase=args.phase,test_evaluated=False))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
