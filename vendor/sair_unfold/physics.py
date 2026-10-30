"""Real image -> complex visibility, whitening, and real-space spectral filters."""
import torch
from torch import nn
from .numerics import cg
from .operators import MatrixSAIROperator

class Physics(nn.Module):
    def __init__(self, operator, config, noise_sigma=None):
        super().__init__()
        self.operator, self.config = operator, config
        device = next(operator.buffers()).device
        sigma = torch.as_tensor(config.noise_sigma if noise_sigma is None else noise_sigma,
                                dtype=torch.float32, device=device)
        if sigma.ndim == 0:
            sigma = sigma.expand(operator.measurements).clone()
        if sigma.shape != (operator.measurements,) or not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
            raise ValueError("noise_sigma must be positive scalar or [M], sqrt(E|n|^2)")
        self.register_buffer("sigma", sigma)
        ones = torch.ones(1, 1, config.image_size, config.image_size, device=device)
        self.register_buffer("offset", operator(ones * config.center).detach())
        # s is an estimate used only for numerical scaling, not a step-size bound.
        rng = torch.Generator(device=device).manual_seed(1234)
        v = torch.randn(ones.shape, device=device, generator=rng)
        v = v / v.norm()
        with torch.no_grad():
            for _ in range(30):
                w = operator.adjoint(operator(v) / sigma.square()) * config.scale**2
                norm = w.norm()
                if not bool(torch.isfinite(norm)):
                    raise ValueError("Non-finite weighted operator")
                if float(norm) == 0:
                    break
                v = w / norm
            s = (operator(v) * config.scale / sigma).abs().square().sum().sqrt()
        if not bool(torch.isfinite(s)) or float(s) <= 0:
            raise ValueError("Operator has zero or invalid sensitivity")
        self.register_buffer("spectral_scale", s)
        self._hard_basis = None

    def forward(self, x):
        return self.operator(x) * self.config.scale / self.sigma / self.spectral_scale

    def adjoint(self, y):
        return self.operator.adjoint(y / self.sigma) * self.config.scale / self.spectral_scale

    def normal(self, x):
        return self.adjoint(self(x))

    def encode(self, visibility):
        if visibility.ndim != 2 or visibility.shape[1] != self.operator.measurements or not torch.is_complex(visibility):
            raise ValueError("Expected complex visibility [B,M]")
        if not bool(torch.isfinite(visibility).all()):
            raise ValueError("Visibility must be finite")
        return (visibility - self.offset) / self.sigma / self.spectral_scale

    def simulate(self, x, generator=None):
        y = self.operator(self.config.denormalize(x))
        re = torch.randn(y.shape, device=y.device, generator=generator)
        im = torch.randn(y.shape, device=y.device, generator=generator)
        return y + self.sigma * torch.complex(re, im) / 2**0.5

    def data_step(self, center, encoded_y, beta):
        rhs = self.adjoint(encoded_y) + beta * center
        return cg(lambda x: self.normal(x) + beta * x, rhs,
                  self.config.cg_iters, self.config.cg_rtol, initial=center)

    def filter(self, correction, mu, mode="soft"):
        if mode == "none":
            return correction, {"iterations": 0}
        if mode == "hard":
            if self._hard_basis is None:
                self._hard_basis = self.hard_basis()
            flat = correction.flatten(1)
            v = self._hard_basis
            return (flat - (flat @ v.T) @ v).reshape_as(correction), {"iterations": 0}
        if mode != "soft":
            raise ValueError("Projection mode must be soft, hard, or none")
        # Q_mu = mu (C* C + mu I)^-1; equivalent to the soft null-space
        # expression after realifying C. Strictly real and matrix-free.
        return cg(lambda x: self.normal(x) + mu * x, mu * correction,
                  self.config.cg_iters, self.config.cg_rtol)

    @torch.no_grad()
    def hard_basis(self):
        if not isinstance(self.operator, MatrixSAIROperator):
            raise ValueError("Hard ablation requires an explicit small matrix")
        g = self.operator.matrix * self.config.scale / self.sigma[:, None] / self.spectral_scale
        if 2 * g.numel() > 2_000_000:
            raise ValueError("Hard SVD ablation limited to 2 million real matrix entries")
        real = torch.cat((g.real, g.imag), dim=0).double()
        _, singular, vh = torch.linalg.svd(real, full_matrices=False)
        tol = max(real.shape) * torch.finfo(real.dtype).eps * singular[0]
        return vh[singular > tol].to(g.real.dtype)

    def descriptor(self):
        return {"operator": self.operator.metadata(), "noise_sigma": self.sigma.cpu().tolist(),
                "spectral_scale": float(self.spectral_scale),
                "image_units": "K", "complex_noise_convention": "E|n_m|^2 = sigma_m^2"}
