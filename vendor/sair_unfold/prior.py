"""VE Gaussian corruption: eps_theta(x+sigma eps, log sigma), one-call denoiser."""
import torch
from torch import nn
from .model import UNet

class DiffusionPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = UNet(config.base_channels, config.channel_mult)

    def epsilon(self, z, sigma):
        sigma = torch.as_tensor(sigma, device=z.device, dtype=z.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(len(z))
        if sigma.shape != (len(z),) or bool((sigma <= 0).any()):
            raise ValueError("Prior noise sigma must be positive scalar or [B]")
        t = (sigma.log() - z.new_tensor(self.config.sigma_min).log()) / (
            z.new_tensor(self.config.sigma_max / self.config.sigma_min).log()) * 1000
        return self.net(z, t)

    def forward(self, z, sigma):
        sigma = torch.as_tensor(sigma, device=z.device, dtype=z.dtype)
        factor = sigma if sigma.ndim == 0 else sigma[:, None, None, None]
        return z - factor * self.epsilon(z, sigma)

    def dsm_loss(self, clean, generator=None):
        u = torch.rand(len(clean), device=clean.device, generator=generator)
        sigma = self.config.sigma_min * (self.config.sigma_max / self.config.sigma_min)**u
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        noisy = clean + sigma[:, None, None, None] * noise
        return (self.epsilon(noisy, sigma) - noise).square().mean()

class CNNPrior(nn.Module):
    """Deterministic residual denoiser baseline; must be trained separately."""
    def __init__(self, config):
        super().__init__()
        c = config.base_channels
        self.net = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, sigma):
        return x + self.net(x)
