"""Batched real SPD conjugate gradients; gradients pass through the iterations."""
import torch

def inner(a, b):
    return (a * b).flatten(1).sum(1).reshape(-1, *([1] * (a.ndim - 1)))

def cg(matvec, rhs, iterations=20, rtol=1e-5, initial=None):
    if iterations < 1 or rtol <= 0:
        raise ValueError("Invalid CG settings")
    x = torch.zeros_like(rhs) if initial is None else initial
    r = rhs - matvec(x)
    p = r
    rr = inner(r, r)
    threshold = inner(rhs, rhs) * rtol**2
    used = 0
    for _ in range(iterations):
        active = rr > threshold
        if not bool(active.any()):
            break
        ap = matvec(p)
        pap = inner(p, ap)
        if bool(((pap <= 0) & active).any()):
            raise FloatingPointError("CG encountered a non-SPD direction")
        # Inactive rows use denominator=1, never a tiny-divisor hidden NaN.
        step = torch.where(active, rr / torch.where(active, pap, torch.ones_like(pap)), 0)
        x = x + step * p
        r_new = r - step * ap
        rr_new = inner(r_new, r_new)
        ratio = torch.where(active, rr_new / torch.where(active, rr, torch.ones_like(rr)), 0)
        p = torch.where(active, r_new + ratio * p, torch.zeros_like(p))
        r, rr = r_new, rr_new
        used += 1
    residual = rhs - matvec(x)
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Non-finite CG output")
    rhs_norm = inner(rhs, rhs).sqrt()
    rel = inner(residual, residual).sqrt() / rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
    rel = torch.where(rhs_norm == 0, inner(residual, residual).sqrt(), rel)
    return x, {"iterations": used, "relative_residual": rel.flatten().detach()}
