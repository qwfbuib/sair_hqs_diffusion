import common
import unittest
import torch
from evaluate_unfold import load_full_model,predict,WITHOUT_PRIOR
from train_native_hqs import Config,HQSUnfold,SCHEMA,CheckedPhysics,seed_all
from sair_unfold.operators import MatrixSAIROperator
import numpy as np


class FullCheckpointTests(unittest.TestCase):
    def test_loads_stage_parameters_and_prior(self):
        cfg=Config(image_size=8,base_channels=4,channel_mult=[1]).validate()
        trained=HQSUnfold(cfg)
        with torch.no_grad():
            trained.log_mu.add_(.3)
            trained.alpha_logits.add_(.7)
        checkpoint=dict(schema=SCHEMA,phase='unfold',identity=dict(model_config=cfg.to_dict(),
                        training=dict(projection='soft')),model=trained.state_dict())
        loaded,_=load_full_model(checkpoint)
        for key,value in trained.state_dict().items():
            torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(loaded.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        checkpoint['model']=dict(checkpoint['model'])
        del checkpoint['model']['log_mu']
        with self.assertRaises(RuntimeError): load_full_model(checkpoint)

    def test_rejects_prior_only_checkpoint(self):
        with self.assertRaises(ValueError):
            load_full_model(dict(schema=SCHEMA,phase='prior'))

    def test_without_prior_is_exactly_data_only_and_never_calls_prior(self):
        seed_all(42)
        cfg=Config(image_size=8,base_channels=4,channel_mult=[1],cg_iters=128).validate()
        model=HQSUnfold(cfg).eval()
        expected_model=HQSUnfold(cfg,prior_kind='none').eval()
        with torch.no_grad():
            model.log_beta.add_(.3)
            for name in ('log_beta','log_mu','sigma_logits','alpha_logits'):
                getattr(expected_model,name).copy_(getattr(model,name))
        class ForbiddenPrior(torch.nn.Module):
            def forward(self,*args): raise AssertionError('Prior was called')
        model.prior=ForbiddenPrior()
        rng=np.random.default_rng(42)
        g=(rng.normal(size=(12,64))+1j*rng.normal(size=(12,64)))/64
        p=CheckedPhysics(MatrixSAIROperator(g,8),cfg)
        y=p.operator(torch.full((1,1,8,8),230.))
        with torch.no_grad():
            expected=expected_model(y,p,trace=False)
            actual,trace=predict(model,y,p,WITHOUT_PRIOR)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(all(r['prior_visibility_change_rms']==0 for r in trace))
        self.assertTrue(all(r['filter_cg_residual']==0 for r in trace))


if __name__=='__main__':unittest.main()
