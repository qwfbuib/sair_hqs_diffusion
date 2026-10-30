import math
import torch
from torch import nn
from .prior import DiffusionPrior, CNNPrior

def geometric(start, end, count):
    return torch.logspace(math.log10(start), math.log10(end), count)

class HQSUnfold(nn.Module):
    def __init__(self, config, prior_kind="diffusion", projection="soft"):
        super().__init__()
        if prior_kind not in {"diffusion", "cnn", "none"} or projection not in {"soft", "hard", "none"}:
            raise ValueError("Invalid prior or projection")
        self.config, self.prior_kind, self.projection = config, prior_kind, projection
        self.prior = DiffusionPrior(config) if prior_kind == "diffusion" else (
            CNNPrior(config) if prior_kind == "cnn" else None)
        self.log_beta = nn.Parameter(geometric(config.beta_start, config.beta_end, config.stages).log())
        self.log_mu = nn.Parameter(geometric(config.mu_start, config.mu_end, config.stages).log())
        sigmas = geometric(config.sigma_start, config.sigma_end, config.stages)
        fraction = (sigmas - config.sigma_min) / (config.sigma_max - config.sigma_min)
        self.sigma_logits = nn.Parameter(torch.logit(fraction))
        self.alpha_logits = nn.Parameter(torch.full((config.stages,), math.log(config.alpha_init / (1-config.alpha_init))))

    def stage_parameters(self):
        # Bounds avoid singular solves and keep prior within trained sigma support.
        beta = self.log_beta.clamp(-11.5, 6).exp()
        mu = self.log_mu.clamp(-11.5, 6).exp()
        sigma = self.config.sigma_min + (self.config.sigma_max-self.config.sigma_min) * self.sigma_logits.sigmoid()
        return beta, mu, sigma, self.alpha_logits.sigmoid()

    def forward(self, visibility, physics, projection=None, disable_prior=False, trace=False):
        y = physics.encode(visibility)
        zero = visibility.real.new_zeros(len(visibility), 1, self.config.image_size, self.config.image_size)
        v, init_info = physics.data_step(zero, y, zero.new_tensor(self.config.initial_beta))
        beta, mu, sigma, alpha = self.stage_parameters()
        records = []
        for k in range(self.config.stages):
            x, dc_info = physics.data_step(v, y, beta[k])
            if self.prior is None or disable_prior:
                correction = torch.zeros_like(x)
                filtered, filter_info = correction, {"iterations": 0}
            else:
                correction = self.prior(x, sigma[k]) - x
                filtered, filter_info = physics.filter(correction, mu[k], projection or self.projection)
            v = x + alpha[k] * filtered
            # No clipping: clipping can destroy null-space consistency.
            if trace:
                with torch.no_grad():
                    before = physics.operator(self.config.denormalize(x)) - visibility
                    after = physics.operator(self.config.denormalize(v)) - visibility
                    records.append({"stage": k, "beta": float(beta[k]), "mu": float(mu[k]),
                        "sigma": float(sigma[k]), "alpha": float(alpha[k]),
                        "dc_cg_residual": float(dc_info["relative_residual"].max()),
                        "filter_cg_residual": float(filter_info.get("relative_residual", torch.tensor([0.])).max()),
                        "visibility_mse_before_prior": float(before.abs().square().mean()),
                        "visibility_mse_after_prior": float(after.abs().square().mean()),
                        "prior_visibility_change_rms": float(physics.operator(filtered).abs().square().mean().sqrt() * self.config.scale * alpha[k])})
        return (v, records) if trace else v
