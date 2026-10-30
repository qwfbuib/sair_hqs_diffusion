"""SS2Dv0-style four-route selective SSM, native PyTorch backend.

Cross routes and the recurrence follow the pinned VMamba reference. The chunked
associative prefix implementation is local, not an official fused VMamba kernel.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def cross_scan(x):
    if x.ndim != 4:
        raise ValueError('Expected NCHW features')
    forward = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
    return torch.cat((forward, forward.flip(-1)), dim=1)


def cross_merge(routes, height, width):
    if routes.ndim != 4 or routes.shape[1] != 4 or routes.shape[-1] != height * width:
        raise ValueError('Expected [B,4,C,H*W]')
    paired = routes[:, :2] + routes[:, 2:].flip(-1)
    column = paired[:, 1].reshape(routes.shape[0], -1, width, height).transpose(2, 3)
    return (paired[:, 0] + column.flatten(2)).reshape(routes.shape[0], -1, height, width)


def selective_scan(u, delta, A, B, C, D, delta_bias, chunk_size=32):
    """h_l = exp(softplus(delta_l+bias)*A) h_(l-1) + dt_l B_l u_l.

    u/delta: [batch,K,inner,L], A: [K,inner,N], B/C: [batch,K,N,L].
    Chunk boundaries carry the full state; they never reset the sequence.
    All recurrence arithmetic is FP32, including under CUDA autocast.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    batch, routes, inner, length = u.shape
    if delta.shape != u.shape or A.shape[:2] != (routes, inner):
        raise ValueError('Invalid scan dimensions')
    if B.shape != C.shape or B.shape != (batch, routes, A.shape[-1], length):
        raise ValueError('Invalid variable B/C dimensions')
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, delta, A, B, C, D = (v.float() for v in (u, delta, A, B, C, D))
        delta = F.softplus(delta + delta_bias.float()[None, :, :, None])
        state = u.new_zeros(batch, routes, inner, A.shape[-1])
        outputs = []
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            dt = delta[..., start:stop]
            a = torch.exp(dt.unsqueeze(-2) * A[None, :, :, :, None])
            b = (dt * u[..., start:stop]).unsqueeze(-2) * B[:, :, None, :, start:stop]
            offset = 1
            while offset < stop - start:
                # Composition (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
                next_b = torch.cat((b[..., :offset], b[..., offset:] + a[..., offset:] * b[..., :-offset]), dim=-1)
                a = torch.cat((a[..., :offset], a[..., offset:] * a[..., :-offset]), dim=-1)
                b = next_b
                offset *= 2
            states = b + a * state.unsqueeze(-1)
            outputs.append((states * C[:, :, None, :, start:stop]).sum(-2))
            state = states[..., -1]
        return torch.cat(outputs, dim=-1) + u * D[None, :, :, None]


class Mamba2DBlock(nn.Module):
    backend = 'torch_associative_scan_v1'

    def __init__(self, channels, time_width=128, d_state=16, expand=2,
                 d_conv=3, chunk_size=32, time_conditioning=False, checkpoint_scan=True):
        super().__init__()
        if channels < 1 or d_state < 1 or expand < 1 or d_conv not in (0, 3):
            raise ValueError('Positive dimensions and d_conv in {0,3} required')
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        self.channels, self.inner, self.d_state = channels, channels * expand, d_state
        self.rank = math.ceil(channels / 16)
        self.chunk_size, self.checkpoint_scan = chunk_size, checkpoint_scan
        self.norm = nn.LayerNorm(channels)
        self.time = nn.Linear(time_width, 2 * channels) if time_conditioning else None
        if self.time is not None:
            nn.init.zeros_(self.time.weight)
            nn.init.zeros_(self.time.bias)
        self.in_proj = nn.Linear(channels, 2 * self.inner, bias=False)
        self.local_conv = (nn.Conv2d(self.inner, self.inner, d_conv, padding=d_conv // 2,
                                    groups=self.inner) if d_conv else nn.Identity())
        x_projections = [nn.Linear(self.inner, self.rank + 2 * d_state, bias=False) for _ in range(4)]
        self.x_proj_weight = nn.Parameter(torch.stack([v.weight for v in x_projections]))
        self.dt_weight = nn.Parameter(torch.empty(4, self.inner, self.rank))
        nn.init.uniform_(self.dt_weight, -self.rank ** -.5, self.rank ** -.5)
        dt = torch.exp(torch.rand(4, self.inner) * (math.log(.1) - math.log(.001)) + math.log(.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log()[None, None].repeat(4, self.inner, 1))
        self.D = nn.Parameter(torch.ones(4, self.inner))
        self.out_norm = nn.LayerNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, channels, bias=False)

    def _core(self, x):
        routes = cross_scan(x)
        projected = torch.einsum('bkdl,kod->bkol', routes, self.x_proj_weight)
        dt, B, C = torch.split(projected, [self.rank, self.d_state, self.d_state], dim=2)
        dt = torch.einsum('bkrl,kdr->bkdl', dt, self.dt_weight)
        y = selective_scan(routes, dt, -self.A_log.float().exp(), B, C,
                           self.D, self.dt_bias, self.chunk_size)
        return cross_merge(y, x.shape[-2], x.shape[-1])

    def forward(self, x, t_emb=None):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Mamba2DBlock expects matching NCHW channels')
        h = self.norm(x.permute(0, 2, 3, 1))
        if self.time is not None:
            if t_emb is None or t_emb.shape != (x.shape[0], self.time.in_features):
                raise ValueError('Time-conditioned block requires [B,time_width]')
            scale, shift = self.time(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, None, None]) + shift[:, None, None]
        h, gate = self.in_proj(h).chunk(2, dim=-1)
        h = F.silu(self.local_conv(h.permute(0, 3, 1, 2).contiguous()))
        # Frozen/eval priors still need input and sigma gradients during HQS.
        if torch.is_grad_enabled() and h.requires_grad and self.checkpoint_scan:
            y = checkpoint(self._core, h, use_reentrant=False, preserve_rng_state=False)
        else:
            y = self._core(h)
        y = self.out_norm(y.permute(0, 2, 3, 1).to(h.dtype)) * F.silu(gate)
        return x + self.out_proj(y).permute(0, 3, 1, 2)
