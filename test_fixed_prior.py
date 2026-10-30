"""Small independent checks for physical solves, normalization and metrics."""
import common
import unittest
import numpy as np
import torch
from infer_fixed_prior import metrics, summarize
from train_native_hqs import CheckedPhysics, Config, seed_all
from sair_unfold.operators import MatrixSAIROperator


class InferenceTests(unittest.TestCase):
    def test_physics_against_dense_real_system(self):
        seed_all(42)
        cfg=Config(image_size=8,base_channels=4,channel_mult=[1],cg_iters=128,
                   t_min=150,t_max=320,noise_sigma=.1).validate()
        rng=np.random.default_rng(42)
        matrix=(rng.normal(size=(12,64))+1j*rng.normal(size=(12,64)))/64
        physics=CheckedPhysics(MatrixSAIROperator(matrix,8),cfg)
        z=torch.randn(1,1,8,8)
        y=physics.operator(cfg.denormalize(z))
        b=physics.encode(y)
        torch.testing.assert_close(physics(z),b,atol=2e-6,rtol=1e-4)
        c=physics.operator.matrix*cfg.scale/physics.sigma[:,None]/physics.spectral_scale
        real=torch.cat([c.real,c.imag]).double()
        obs=torch.cat([b.real,b.imag],dim=1).double().flatten()
        center=torch.randn_like(z); beta=torch.tensor(.1)
        actual,_=physics.data_step(center,b,beta)
        expected=torch.linalg.solve(real.T@real+beta*torch.eye(64,dtype=torch.double),
                                    real.T@obs+beta*center.double().flatten())
        torch.testing.assert_close(actual.double().flatten(),expected,atol=3e-5,rtol=3e-5)
        direction=torch.randn_like(z); mu=torch.tensor(.07)
        actual,_=physics.filter(direction,mu,'soft')
        expected=torch.linalg.solve(real.T@real+mu*torch.eye(64,dtype=torch.double),
                                    mu*direction.double().flatten())
        torch.testing.assert_close(actual.double().flatten(),expected,atol=3e-5,rtol=3e-5)
        self.assertLessEqual(max(physics.cg_records),cfg.cg_rtol)

    def test_metrics_and_failed_results(self):
        truth=torch.full((1,1,16,16),200.)
        residual=torch.full((1,4),2+0j)
        same=metrics(truth,truth,torch.zeros_like(residual),2.)
        self.assertAlmostEqual(same['ssim'],1.)
        self.assertEqual(same['rmse_k'],0.)
        self.assertIsNone(same['psnr_db'])
        shifted=metrics(truth+3,truth,residual,2.)
        self.assertEqual(shifted['rmse_k'],3.)
        self.assertEqual(shifted['bias_k'],3.)
        self.assertEqual(shifted['standardized_residual_energy'],1.)
        summary=summarize([dict(mode='prior_soft',status='failed')])
        self.assertFalse(summary['prior_soft']['complete'])
        self.assertEqual(summary['prior_soft']['failed'],1)


if __name__=='__main__':
    unittest.main()
