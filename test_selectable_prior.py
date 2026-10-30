import common
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from selectable_prior.ss2d import cross_scan,cross_merge,selective_scan,Mamba2DBlock
from selectable_prior.reference.vmamba_reference import cross_scan_fwd,cross_merge_fwd,selective_scan_torch
from selectable_prior.models import SelectablePrior,SelectableHQS,architecture
from selectable_prior.checkpoints import SCHEMA,load_model
from selectable_prior.training import train_phase
from train_native_hqs import Config,CheckedPhysics,SCHEMA as LEGACY_SCHEMA,seed_all
from sair_unfold.prior import DiffusionPrior
from sair_unfold.unfolding import HQSUnfold
from sair_unfold.operators import MatrixSAIROperator


def tiny_config():
    return Config(image_size=8,base_channels=4,channel_mult=[1,2,4,4],
                  stages=8,cg_iters=128,t_min=150,t_max=320,noise_sigma=.1).validate()


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class ScanTests(unittest.TestCase):
    def setUp(self):seed_all(42)

    def test_routes_and_cross_chunk_reference_all_gradients(self):
        x=torch.randn(2,3,5,7,requires_grad=True)
        torch.testing.assert_close(cross_scan(x),cross_scan_fwd(x),rtol=0,atol=0)
        routes=torch.randn(2,4,3,35,requires_grad=True)
        merged=cross_merge(routes,5,7)
        ref=cross_merge_fwd(routes.reshape(2,4,3,5,7)).reshape_as(merged)
        torch.testing.assert_close(merged,ref,rtol=0,atol=0)
        a=torch.autograd.grad(merged.square().sum(),routes,retain_graph=True)[0]
        b=torch.autograd.grad(ref.square().sum(),routes)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        raw=[torch.randn(2,4,3,65),torch.randn(2,4,3,65),-torch.rand(4,3,4)-.2,
             torch.randn(2,4,4,65),torch.randn(2,4,4,65),torch.randn(4,3),torch.randn(4,3)-2]
        inputs=[v.clone().requires_grad_() for v in raw]
        reference=[v.clone().requires_grad_() for v in raw]
        u,dt,A,B,C,D,bias=reference
        expected=selective_scan_torch(u.flatten(1,2),dt.flatten(1,2),A.flatten(0,1),B,C,
                                      D.flatten(),delta_bias=bias.flatten()).reshape_as(u)
        for chunk in (1,7,32,128):
            torch.testing.assert_close(selective_scan(*inputs,chunk_size=chunk),expected,atol=3e-5,rtol=3e-5)
        weight=torch.randn_like(expected)
        actual_grads=torch.autograd.grad((selective_scan(*inputs)*weight).sum(),inputs)
        ref_grads=torch.autograd.grad((expected*weight).sum(),reference)
        for a,b in zip(actual_grads,ref_grads):torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-5)

    def test_frozen_eval_checkpoint_recomputes_with_matching_input_and_time_gradients(self):
        block=Mamba2DBlock(8,time_width=16,d_state=4,time_conditioning=True,checkpoint_scan=False)
        with torch.no_grad():block.time.weight.normal_(std=.01)
        block.eval().requires_grad_(False)
        x=torch.randn(1,8,3,5,requires_grad=True);t=torch.randn(1,16,requires_grad=True)
        original=block(x,t);g=torch.randn_like(original)
        gradients=torch.autograd.grad((original*g).sum(),(x,t))
        block.checkpoint_scan=True
        from selectable_prior import ss2d
        with patch.object(ss2d,'checkpoint',wraps=ss2d.checkpoint) as recompute:
            actual=block(x,t)
            compared=torch.autograd.grad((actual*g).sum(),(x,t))
            self.assertGreater(recompute.call_count,0)
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
        for a,b in zip(gradients,compared):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(bool(torch.isfinite(b).all()))
            self.assertGreater(float(b.abs().sum()),0)

    def test_chunk_boundary_carries_state(self):
        u=torch.ones(1,4,1,65)
        out=selective_scan(u,torch.zeros_like(u),torch.zeros(4,1,1),
            torch.ones(1,4,1,65),torch.ones(1,4,1,65),torch.zeros(4,1),torch.zeros(4,1),32)
        torch.testing.assert_close(out[0,0,0],torch.arange(1,66)*torch.nn.functional.softplus(torch.tensor(0.)))


class ModelTests(unittest.TestCase):
    def setUp(self):seed_all(42)

    def test_architectures_interfaces_sigma_and_same_training_noise(self):
        cfg=tiny_config();clean=torch.randn(1,1,8,8)
        draws=[]
        for backbone in ('unet','mamba'):
            model=SelectablePrior(cfg,backbone)
            capture=[]
            hook=model.net.register_forward_pre_hook(lambda _,args:capture.append(tuple(v.detach().clone() for v in args)))
            loss=model.dsm_loss(clean,torch.Generator().manual_seed(123))
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward();draws.append(capture[0]);hook.remove()
            with torch.no_grad():model.net.output[-1].weight.normal_(std=.02)
            result=model(clean,.1)
            self.assertEqual(result.shape,clean.shape)
            self.assertFalse(torch.equal(model.epsilon(clean,.1),model.epsilon(clean,.3)))
            model.eval().requires_grad_(False)
            x=clean.clone().requires_grad_();sigma=torch.tensor(.1,requires_grad=True)
            gradients=torch.autograd.grad(model(x,sigma).square().mean(),(x,sigma))
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        for a,b in zip(draws[0],draws[1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError):SelectablePrior(cfg,'invalid')

    def test_legacy_unet_and_full_hqs_are_exactly_compatible(self):
        cfg=tiny_config()
        legacy=DiffusionPrior(cfg)
        with torch.no_grad():legacy.net.output[-1].weight.normal_(std=.01)
        identity=dict(model_config=cfg.to_dict(),training=dict(projection='soft'),data_core={'data':'x'})
        state=dict(schema=LEGACY_SCHEMA,phase='prior',identity=identity,model=legacy.state_dict())
        loaded,_,_=load_model(state,expected_backbone='unet')
        x=torch.randn(1,1,8,8)
        torch.testing.assert_close(legacy(x,.2),loaded(x,.2),rtol=0,atol=0)
        with self.assertRaises(ValueError):load_model(state,expected_backbone='mamba')
        with self.assertRaises(ValueError):load_model(state,data_core={'data':'wrong'})
        full=HQSUnfold(cfg);full.prior.load_state_dict(legacy.state_dict())
        state=dict(schema=LEGACY_SCHEMA,phase='unfold',identity=identity,model=full.state_dict())
        loaded,_,_=load_model(state)
        op=MatrixSAIROperator(np.random.default_rng(2).normal(size=(12,64)).astype(complex)/64,8)
        p=CheckedPhysics(op,cfg);y=op(torch.full((1,1,8,8),220.))
        torch.testing.assert_close(full(y,p),loaded(y,p),rtol=0,atol=0)

    def test_new_identity_and_strict_state_checks(self):
        cfg=tiny_config();model=SelectablePrior(cfg,'mamba')
        identity=dict(model_config=cfg.to_dict(),architecture=architecture('mamba'),data_core={'data':'x'})
        state=dict(schema=SCHEMA,phase='prior',identity=identity,model=model.state_dict())
        loaded,_,_=load_model(state,expected_backbone='mamba')
        self.assertIsInstance(loaded,SelectablePrior)
        with self.assertRaises(ValueError):load_model(state,expected_backbone='unet')
        with self.assertRaises(ValueError):load_model(state,expected_phase='unfold')
        bad=copy.deepcopy(state);bad['identity']['architecture']['scan_backend']='other'
        with self.assertRaises(ValueError):load_model(bad)
        bad=copy.deepcopy(state);del bad['model']['net.encoder.2.1.A_log']
        with self.assertRaises(RuntimeError):load_model(bad)


class FakeData:
    def __init__(self,root):
        self.root=Path(root);self.identity={'fixture':'fixed-small-data'}
        self.train=['train/a','train/b'];self.selected_val=['val/a']
    def batch(self,scenes,device,gamma=None):
        base=torch.linspace(180.,270.,64).reshape(1,1,8,8)
        target=torch.cat([base+(i.endswith('b')) for i in scenes]).to(device)
        if gamma is None:return target
        p=fake_bank(self,tiny_config(),[gamma],device)[gamma]
        return target,p.operator(target).detach(),.1


def fake_bank(data,cfg,gammas,device,*args,**kw):
    matrix=(np.random.default_rng(42).normal(size=(12,64))+1j*np.random.default_rng(43).normal(size=(12,64)))/64
    return {g:CheckedPhysics(MatrixSAIROperator(matrix,8).to(device),cfg) for g in gammas}


class ResumeTests(unittest.TestCase):
    def test_resume_to_missing_run_is_rejected_without_creating_output(self):
        from selectable_prior.training import main
        config=json.loads((common.ROOT/'configs/server_128_unet.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json';path.write_text(json.dumps(config))
            data=FakeData(Path(folder)/'data')
            output=Path(folder)/'missing-run'
            args=SimpleNamespace(config=str(path),data=str(data.root),out=str(output),
                device='cpu',prior=None,resume=True,stop_after=None,smoke=False,phase='pipeline')
            with patch('selectable_prior.training.NativeData',return_value=data):
                with self.assertRaisesRegex(ValueError,'existing.*last.pt'):main(args)
            self.assertFalse(output.exists())

    def test_skipped_update_replays_the_same_samples_and_noise(self):
        from selectable_prior import training
        cfg=tiny_config()
        config=json.loads((common.ROOT/'configs/server_128_unet.json').read_text())
        config['model']=cfg.to_dict()
        config['training'].update(prior_updates=2,prior_batch_size=1,prior_amp=False,
            prior_validate_every=1,checkpoint_every=1,log_every=1,val_scenes_per_class=1)
        original_update=training.optimizer_update
        calls=[]
        def skip_first(*args,**kw):
            calls.append(1)
            return (0.,True) if len(calls)==1 else original_update(*args,**kw)
        with tempfile.TemporaryDirectory() as folder:
            data=FakeData(folder)
            a=SimpleNamespace(out=str(Path(folder)/'normal'),device='cpu',prior=None,resume=False,stop_after=None)
            b=SimpleNamespace(**vars(a));b.out=str(Path(folder)/'retry')
            train_phase('prior',a,config,data)
            with patch('selectable_prior.training.optimizer_update',side_effect=skip_first):
                train_phase('prior',b,config,data)
            normal=torch.load(Path(a.out)/'prior/last.pt',weights_only=False)
            retried=torch.load(Path(b.out)/'prior/last.pt',weights_only=False)
            self.assertEqual(len(calls),3)
            for key in ('model','optimizer','scaler','rng','noise_rng','sample_position','step','best','identity'):
                self.assertTrue(equal(normal[key],retried[key]),key)

    def test_both_backbones_both_phases_resume_exactly(self):
        cfg=tiny_config()
        original=json.loads((common.ROOT/'configs/server_128_unet.json').read_text())
        t=original['training'];t.update(prior_updates=2,unfold_updates=2,prior_batch_size=1,
            unfold_batch_size=1,unfold_accumulation=1,prior_amp=False,prior_validate_every=1,
            unfold_validate_every=1,checkpoint_every=1,log_every=1,val_scenes_per_class=1,
            gammas=[1.])
        with tempfile.TemporaryDirectory() as folder:
            data=FakeData(folder)
            with patch('selectable_prior.training.physics_bank',side_effect=fake_bank):
                for backbone in ('unet','mamba'):
                    config=dict(model=cfg.to_dict(),training=t,prior={'backbone':backbone})
                    prior_path=None
                    for phase in ('prior','unfold'):
                        full=Path(folder)/(backbone+phase+'full')
                        partial=Path(folder)/(backbone+phase+'partial')
                        def args(out,resume=False,stop=None):
                            return SimpleNamespace(out=str(out),device='cpu',prior=prior_path,
                                                   resume=resume,stop_after=stop)
                        self.assertTrue(train_phase(phase,args(full),config,data))
                        self.assertFalse(train_phase(phase,args(partial,stop=1),config,data))
                        self.assertTrue(train_phase(phase,args(partial,resume=True),config,data))
                        a=torch.load(full/phase/'last.pt',weights_only=False)
                        b=torch.load(partial/phase/'last.pt',weights_only=False)
                        for key in ('model','optimizer','scaler','rng','noise_rng','sample_position','step','best','identity'):
                            self.assertTrue(equal(a[key],b[key]),(backbone,phase,key))
                        prior_path=str(full/'prior/best.pt') if phase=='prior' else prior_path
                        wrong=copy.deepcopy(config);wrong['prior']={'backbone':'mamba' if backbone=='unet' else 'unet'}
                        with self.assertRaises(ValueError):train_phase(phase,args(partial,resume=True),wrong,data)


if __name__=='__main__':unittest.main()
