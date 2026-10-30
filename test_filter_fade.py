import common
import unittest
import numpy as np
import torch
from verify_filter_fade import reconstruct,mixed_correction,WEIGHTS
from train_native_hqs import Config,CheckedPhysics,HQSUnfold,seed_all
from sair_unfold.operators import MatrixSAIROperator


class FilterFadeTests(unittest.TestCase):
    def test_mixture_direction_and_endpoints(self):
        d=torch.tensor([1.,-2.]);q=d*.2
        self.assertTrue(torch.equal(mixed_correction(d,q,1),q))
        self.assertTrue(torch.equal(mixed_correction(d,q,0),d))
        torch.testing.assert_close(mixed_correction(d,q,.25),.8*d)
        self.assertEqual(WEIGHTS,(1.,1.,1.,1.,.75,.5,.25,0.))
        with self.assertRaises(ValueError):mixed_correction(d,q,-1)

    def test_full_trajectory_matches_existing_endpoints(self):
        seed_all(20261023)
        cfg=Config(image_size=8,base_channels=4,channel_mult=[1],stages=8,cg_iters=128,
                   t_min=150,t_max=320,noise_sigma=.1).validate()
        rng=np.random.default_rng(42)
        g=(rng.normal(size=(12,64))+1j*rng.normal(size=(12,64)))/64
        p=CheckedPhysics(MatrixSAIROperator(g,8),cfg)
        model=HQSUnfold(cfg).eval()
        y=p.operator(torch.full((1,1,8,8),230.))
        with torch.no_grad():
            for weight,mode in [(1.,'soft'),(0.,'none')]:
                expected=model(y,p,projection=mode,trace=False)
                actual,_=reconstruct(model,p,y,(weight,)*8,trace=False)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            result,trace=reconstruct(model,p,y)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual([r['filter_weight'] for r in trace],list(WEIGHTS))
        self.assertEqual(trace[-1]['filter_cg_residual'],0.)


if __name__=='__main__':unittest.main()
