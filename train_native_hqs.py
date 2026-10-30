"""Two-stage HQS training on immutable stored native128 observations.

Prior: VE epsilon prediction on train GT only. HQS: frozen pretrained prior,
learn 8-stage beta/mu/sigma/alpha using the stored current gamma y, never
Physics.simulate. Checkpoints bind config, core input hashes and code hashes.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import common
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from sair_dataset.reader import DatasetReader
from sair_experiments.operators import dataset_operator
from sair_unfold.config import Config
from sair_unfold.prior import DiffusionPrior
from sair_unfold.unfolding import HQSUnfold
from sair_unfold.physics import Physics
from sair_unfold.numerics import cg
from sair_unfold.losses import reconstruction_loss
from unfold_bridge import Native128YOperator

SCHEMA='native128-hqs-training-v1'

def replace_with_retry(tmp,path,timeout=2.):
    """Windows readers may temporarily deny rename/delete sharing."""
    deadline=time.monotonic()+timeout
    while True:
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.05)

def atomic_json(path,obj,required=True):
    path=Path(path);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    try:
        common.write_json(tmp,obj)
        replace_with_retry(tmp,path,timeout=2. if required else .5)
        return True
    except OSError as exc:
        if required:raise
        print(f'Progress snapshot unavailable; training continues: {path.name}: {exc}',file=sys.stderr,flush=True)
        return False
    finally:
        try:tmp.unlink(missing_ok=True)
        except OSError:pass

def atomic_save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    torch.save(obj,tmp);replace_with_retry(tmp,path,timeout=5.)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])

def optimizer_update(optimizer,scaler,parameters,clip,use_amp):
    scaler.unscale_(optimizer)
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,clip,error_if_nonfinite=not use_amp))
    previous=scaler.get_scale()
    scaler.step(optimizer);scaler.update()
    skipped=bool(use_amp and scaler.get_scale()<previous)
    return norm,skipped

def code_hashes():
    paths=[common.ROOT/n for n in ('train_native_hqs.py','common.py','torch_operator.py','unfold_bridge.py')]
    for name in ('sair_dataset','sair_experiments','sair_unfold'):
        paths.extend(sorted((common.ROOT/'vendor'/name).glob('*.py')))
    return {str(p.relative_to(common.ROOT)):common.sha(p) for p in paths}

class NativeData:
    def __init__(self,path,val_per_class):
        self.root=Path(path).resolve();self.reader=DatasetReader(self.root)
        m=self.reader.metadata;c=m['config']
        if not (m['n_grid']==128 and m['bt_min']==150 and m['bt_max']==320 and
                c['n_arm']==21 and c['spacing_lambda']==.875 and c['theta']==1.05):
            raise ValueError('Expected audited native128 Y-array, 150..320 K')
        for name in ('audit_report.json','visibility_verification.json','native128_generation_report.json'):
            if json.loads((self.root/name).read_text(encoding='utf-8'))['status']!='pass':
                raise ValueError('Dataset acceptance not pass: '+name)
        self.train=sorted(s for s in self.reader.scenes if m['class_assignment'][s.split('/')[0]]=='train')
        self.val=sorted(s for s in self.reader.scenes if m['class_assignment'][s.split('/')[0]]=='val')
        if (len(self.train),len(self.val))!=(21700,4900):raise ValueError('Wrong train/val counts')
        self.selected_val=[]
        for cls in sorted(c for c,s in m['class_assignment'].items() if s=='val'):
            scenes=[s for s in self.val if s.startswith(cls+'/')]
            scenes.sort(key=lambda s:hashlib.sha256(('20261023/'+s).encode()).hexdigest())
            self.selected_val.extend(scenes[:val_per_class])
        self.cache=OrderedDict()
        self.identity={name:common.sha(self.root/name) for name in
                       ('scenes.json','geometry.npz','summary.csv','visibility_verification.json')}

    def truth(self,scene):
        # No test GT or measurements can be read by this training interface.
        split=self.reader.metadata['class_assignment'][scene.split('/')[0]]
        if split not in ('train','val'):raise ValueError('Training cannot read test')
        if scene not in self.cache:
            u8=np.load(self.root/'gt'/(scene+'.npy'),allow_pickle=False)
            if u8.shape!=(128,128) or u8.dtype!=np.uint8:raise ValueError('Invalid native128 GT')
            self.cache[scene]=((150.+170.*u8.astype(np.float64)/255.).astype(np.float32))[None]
            if len(self.cache)>256:self.cache.popitem(last=False)
        self.cache.move_to_end(scene)
        return self.cache[scene]

    def batch(self,scenes,device,gamma=None):
        T=torch.from_numpy(np.stack([self.truth(s) for s in scenes])).to(device)
        if gamma is None:return T
        obs=[self.reader.load(s,gamma,repeat=0) for s in scenes]
        sig=np.array([s['sigma'] for s in obs])
        if not np.all(sig==sig[0]) or sig[0]<=0:raise ValueError('Unexpected noise scales')
        y=torch.from_numpy(np.stack([s['y'] for s in obs])).to(device)
        return T,y,float(sig[0])

class Cursor:
    """Deterministic epoch shuffle whose exact position is checkpointed."""
    def __init__(self,size,seed,position=0):
        self.size,self.seed,self.position=size,seed,position
        self.epoch=-1;self.order=None
    def take(self,n):
        result=[]
        while len(result)<n:
            ep,offset=divmod(self.position,self.size)
            if ep!=self.epoch:
                self.order=np.random.default_rng(np.random.SeedSequence([self.seed,ep])).permutation(self.size)
                self.epoch=ep
            count=min(n-len(result),self.size-offset)
            result.extend(self.order[offset:offset+count].tolist());self.position+=count
        return result

class CheckedPhysics(Physics):
    """Track initial and all stage solves; certify true residual, not CG recurrence."""
    def __init__(self,*args,inner_factor=.25,**kwargs):
        super().__init__(*args,**kwargs)
        self.inner_factor=inner_factor;self.cg_records=[]
    def _solve(self,matvec,rhs,initial=None):
        x,info=cg(matvec,rhs,self.config.cg_iters,self.config.cg_rtol*self.inner_factor,initial=initial)
        if float(info['relative_residual'].max())>self.config.cg_rtol:
            # Residual replacement/refinement, retaining the autograd graph.
            x,refine=cg(matvec,rhs,self.config.cg_iters,self.config.cg_rtol*self.inner_factor,initial=x)
            info=dict(iterations=info['iterations']+refine['iterations'],relative_residual=refine['relative_residual'])
        value=float(info['relative_residual'].max())
        self.cg_records.append(value)
        if value>self.config.cg_rtol:raise RuntimeError(f'True CG residual {value} exceeds {self.config.cg_rtol}')
        return x,info
    def data_step(self,center,encoded_y,beta):
        rhs=self.adjoint(encoded_y)+beta*center
        return self._solve(lambda x:self.normal(x)+beta*x,rhs,initial=center)
    def filter(self,correction,mu,mode='soft'):
        if mode!='soft':return super().filter(correction,mu,mode)
        return self._solve(lambda x:self.normal(x)+mu*x,mu*correction)

def physics_bank(data,cfg,gammas,device,inner_factor=.25):
    bank={}
    for gamma in gammas:
        op=Native128YOperator(dataset_operator(data.reader,gamma),device=device)
        sigma=float(data.reader.load(data.train[0],gamma)['sigma'])
        bank[gamma]=CheckedPhysics(op,cfg,noise_sigma=sigma,inner_factor=inner_factor)
    return bank

@torch.no_grad()
def validate(model,phase,data,cfg,t,device,bank):
    model.eval();rows=[]
    if phase=='prior':
        generator=torch.Generator(device=device).manual_seed(cfg.seed+9000)
        n=t['prior_batch_size']
        for offset in range(0,len(data.selected_val),n):
            scenes=data.selected_val[offset:offset+n]
            target=cfg.normalize(data.batch(scenes,device))
            value=float(model.dsm_loss(target,generator))
            rows.extend(dict(scene=s,gamma=None,dsm=value) for s in scenes)
        return dict(selection=float(np.mean([r['dsm'] for r in rows])),metric='dsm',scenes=len(rows))
    for scene in data.selected_val:
        for gamma in t['gammas']:
            target,y,sigma=data.batch([scene],device,gamma)
            bank[gamma].cg_records=[]
            predicted,trace=model(y,bank[gamma],trace=True)
            error=cfg.denormalize(predicted)-target
            residual=bank[gamma].operator(cfg.denormalize(predicted))-y
            worst=max(bank[gamma].cg_records)
            rows.append(dict(scene=scene,gamma=gamma,rmse_k=float(error.square().mean().sqrt()),
                             residual_energy=float((residual.abs()/sigma).square().mean()),
                             cg_failed=worst>cfg.cg_rtol,max_cg_relative_residual=worst))
    grouped={}
    for r in rows:grouped.setdefault((r['scene'].split('/')[0],r['gamma']),[]).append(r['rmse_k'])
    return dict(selection=float(np.mean([np.mean(v) for v in grouped.values()])),metric='class_gamma_equal_rmse_k',
                observations=len(rows),scenes=len(data.selected_val),
                cg_failure_rate=float(np.mean([r['cg_failed'] for r in rows])),
                max_cg_relative_residual=max(r['max_cg_relative_residual'] for r in rows),
                by_gamma={str(g):dict(rmse_k=float(np.mean([r['rmse_k'] for r in rows if r['gamma']==g])),
                    residual_energy=float(np.mean([r['residual_energy'] for r in rows if r['gamma']==g]))) for g in t['gammas']})

def run_phase(phase,args,config,data):
    cfg=Config(**config['model']).validate();t=config['training'];device=torch.device(args.device)
    seed=cfg.seed+(100 if phase=='unfold' else 0);seed_all(seed)
    output=Path(args.out).resolve();dest=output/phase
    if dest.exists() and not args.resume:raise FileExistsError('Use a new run or explicit --resume')
    dest.mkdir(parents=True,exist_ok=True)
    bank=physics_bank(data,cfg,t['gammas'],device,t.get('cg_inner_tolerance_factor',.25)) if phase=='unfold' else None
    model=(DiffusionPrior(cfg) if phase=='prior' else HQSUnfold(cfg,'diffusion',t['projection'])).to(device)
    prior_sha=None
    if phase=='unfold':
        prior_path=Path(args.prior) if args.prior else output/'prior'/'best.pt'
        prior=torch.load(prior_path,map_location='cpu',weights_only=False)
        if prior.get('schema')!=SCHEMA or prior.get('phase')!='prior' or prior['identity']['model_config']!=cfg.to_dict():
            raise ValueError('Wrong native128 pretrained prior')
        if prior['identity']['data_core']!=data.identity:raise ValueError('Prior trained on different data')
        model.prior.load_state_dict(prior['model']);prior_sha=common.sha(prior_path)
        if t['freeze_prior']:model.prior.requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg.learning_rate,weight_decay=0.)
    use_amp=phase=='prior' and t['prior_amp'] and device.type=='cuda'
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    identity=dict(model_config=cfg.to_dict(),training=t,data_root=str(data.root),data_core=data.identity,
                  phase=phase,prior_sha256=prior_sha,code=code_hashes())
    step=position=0;best=math.inf
    if args.resume and (dest/'last.pt').exists():
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        if state.get('schema')!=SCHEMA or state['identity']!=identity:raise ValueError('Resume identity differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);step=state['step'];position=state['sample_position'];best=state['best']
        restore_rng(state['rng'])
    total=t[phase+'_updates']
    cursor=Cursor(len(data.train)*(len(t['gammas']) if phase=='unfold' else 1),seed,position)
    atomic_json(dest/'identity.json',identity)
    atomic_json(dest/'environment.json',dict(executable=sys.executable,python=sys.version,torch=torch.__version__,
        numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
        amp=use_amp,train_scenes=len(data.train),val_scene_ids=data.selected_val,test_access=False,
        input_y='stored native128 meas only; no resimulation',normalization='x=(T-235)/85'))
    started=time.perf_counter();start_step=step
    print(f'{phase}: start {step}/{total}, trainable={sum(p.numel() for p in parameters):,}, device={device}',flush=True)

    def save(name):
        atomic_save(dest/name,dict(schema=SCHEMA,phase=phase,identity=identity,model=model.state_dict(),
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),step=step,sample_position=cursor.position,
            best=best,rng=rng_state()))

    last_status_write=0.
    def status(state,**extra):
        nonlocal last_status_write
        now=time.monotonic()
        if state=='running' and now-last_status_write<5.:return
        last_status_write=now
        atomic_json(output/'status.json',dict(status=state,phase=phase,step=step,total_steps=total,
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),
            elapsed_this_session_s=time.perf_counter()-started,best_validation=None if math.isinf(best) else best,
            data=str(data.root),**extra),required=False)

    if step>=total:return True
    with (dest/'metrics.jsonl').open('a',encoding='utf-8') as log:
        while step<total:
            before=time.perf_counter();model.train()
            if phase=='unfold' and t['freeze_prior']:model.prior.eval()
            optimizer.zero_grad(set_to_none=True)
            acc=t['unfold_accumulation'] if phase=='unfold' else 1
            loss_sum=0.;gamma_values=[];worst_cg=0.
            for _ in range(acc):
                batch=t[phase+'_batch_size']
                chosen=cursor.take(batch)
                if phase=='prior':
                    scenes=[data.train[i] for i in chosen]
                    target=cfg.normalize(data.batch(scenes,device))
                    with torch.autocast(device_type=device.type,enabled=use_amp):loss=model.dsm_loss(target)
                else:
                    # Current config uses batch1; arbitrary gamma rows never padded into likelihood.
                    if batch!=1:raise ValueError('HQS native gamma grouping currently requires batch1')
                    si,gi=divmod(chosen[0],len(t['gammas']));gamma=t['gammas'][gi]
                    target,y,sigma=data.batch([data.train[si]],device,gamma)
                    bank[gamma].cg_records=[]
                    predicted,trace=model(y,bank[gamma],trace=True)
                    loss,_=reconstruction_loss(predicted,cfg.normalize(target),y,bank[gamma],cfg)
                    gamma_values.append(gamma)
                    worst_cg=max(worst_cg,max(bank[gamma].cg_records))
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
                loss_sum+=float(loss.detach())/acc
                scaler.scale(loss/acc).backward()
            grad_norm,skipped=optimizer_update(optimizer,scaler,parameters,t['gradient_clip'],use_amp)
            if skipped:
                # An expected AMP overflow reduces the scale; it is not an optimizer update.
                log.write(json.dumps(dict(event='amp_overflow_skipped',optimizer_step=step,
                    sample_position=cursor.position,loss=loss_sum,new_scale=scaler.get_scale()))+'\n');log.flush()
                status('running',event='amp_overflow_skipped',optimizer_step=step)
                continue
            step+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            step_seconds=time.perf_counter()-before
            val=None
            if step==1 or step%t[phase+'_validate_every']==0 or step==total:
                val=validate(model,phase,data,cfg,t,device,bank)
                if val['selection']<best:
                    best=val['selection'];save('best.pt')
                atomic_json(dest/'latest_validation.json',dict(step=step,**val),required=False)
            if step==1 or step%t['checkpoint_every']==0 or val is not None or (args.stop_after and step>=args.stop_after):save('last.pt')
            row=dict(step=step,loss=loss_sum,gradient_norm=grad_norm,step_seconds=step_seconds,
                sample_position=cursor.position,epoch_fraction=cursor.position/cursor.size,
                validation=val,gammas=gamma_values,max_cg_relative_residual=worst_cg,
                cg_failed=worst_cg>cfg.cg_rtol,
                peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            status('running',last_loss=loss_sum,last_step_seconds=step_seconds,
                   optimizer_updates_per_second=(step-start_step)/max(time.perf_counter()-started,1e-9),
                   last_checkpoint_step=step if (dest/'last.pt').exists() and (step==1 or step%t['checkpoint_every']==0 or val is not None) else None)
            if step==1 or step%t['log_every']==0 or val:
                print(f'{phase}: {step}/{total}, loss={loss_sum:.6f}, step_s={step_seconds:.3f}, val={val}, cg={worst_cg:.3g}',flush=True)
            if args.stop_after and step>=args.stop_after:
                status('controlled_stop_for_verification',last_loss=loss_sum);return step>=total
    save('last.pt');status('phase_complete')
    return True

def main(args):
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['training']['test_evaluation']:raise ValueError('Training does not evaluate test')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    data=NativeData(config['data'],config['training']['val_scenes_per_class'])
    out=Path(args.out).resolve()
    if out==data.root or out.is_relative_to(data.root) or data.root.is_relative_to(out):raise ValueError('Output overlaps data')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_config.json').exists() and json.loads((out/'run_config.json').read_text())!=config:
        raise ValueError('Run directory belongs to a different config')
    atomic_json(out/'run_config.json',config)
    phases=('prior','unfold') if args.phase=='pipeline' else (args.phase,)
    try:
        for phase in phases:
            if not run_phase(phase,args,config,data):return
        atomic_json(out/'status.json',dict(status='complete',phase=args.phase,pid=os.getpid(),
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),data=str(data.root),test_evaluated=False),required=False)
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()),required=False)
        atomic_json(out/'status.json',dict(status='failed',error=repr(exc),phase=args.phase,pid=os.getpid()),required=False)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--phase',choices=['pipeline','prior','unfold'],default='pipeline')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--prior');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    main(p.parse_args())
