import math
import torch
from torch.nn import functional as F

def reconstruction_loss(predicted, target, visibility, physics, config):
    image = F.mse_loss(predicted, target)
    residual = physics.operator(config.denormalize(predicted)) - visibility
    likelihood = (residual / physics.sigma).abs().square().mean()
    grad = (F.l1_loss(predicted[..., 1:, :]-predicted[..., :-1, :], target[..., 1:, :]-target[..., :-1, :])
            + F.l1_loss(predicted[..., :, 1:]-predicted[..., :, :-1], target[..., :, 1:]-target[..., :, :-1]))
    bounds = F.relu(predicted.abs()-1).square().mean()
    loss = image + config.visibility_weight * likelihood + config.gradient_weight * grad + config.range_weight * bounds
    return loss, {"image_mse": float(image.detach()), "noise_normalized_visibility_mse": float(likelihood.detach()),
                  "gradient_l1": float(grad.detach()), "range_penalty": float(bounds.detach())}

@torch.no_grad()
def metrics(predicted, target, visibility, physics, config):
    t = config.denormalize(predicted)
    residual = physics.operator(t) - visibility
    result = {"visibility_mse": float(residual.abs().square().mean()),
              "noise_normalized_visibility_mse": float((residual / physics.sigma).abs().square().mean()),
              "out_of_range_fraction": float((predicted.abs() > 1).float().mean())}
    if target is not None:
        error = t - config.denormalize(target)
        mse = float(error.square().mean())
        result.update({"rmse_k": mse**0.5, "mae_k": float(error.abs().mean()),
                       "bias_k": float(error.mean()),
                       "psnr_db": 10*math.log10((config.t_max-config.t_min)**2/max(mse, 1e-20))})
    return result
