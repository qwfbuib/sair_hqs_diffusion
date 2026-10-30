"""Box-constrained GPCG and PDHG with computable convex certificates.

Warm states are bound to one observation, noise repeat, geometry, bounds,
and method. A changed lambda is allowed; a changed observation is not.
"""

from dataclasses import dataclass, field
import hashlib
import time
import numpy as np
from .operators import gradient, gradient_adjoint

FINAL_CERTIFICATE_CALLS = 2


class CallLimit(RuntimeError):
    pass


class Problem:
    def __init__(self, op, y, sigma_ref, bounds=(150., 320.), task_id=""):
        self.op, self.y = op, np.asarray(y, complex)
        self.lo, self.hi = map(float, bounds)
        if self.y.shape != op.u.shape or not np.isfinite(self.y).all():
            raise ValueError("invalid observations")
        if not np.isfinite(sigma_ref) or sigma_ref <= 0 or self.hi <= self.lo:
            raise ValueError("positive physical sigma and increasing bounds required")
        self.delta = self.hi-self.lo
        self.sigma_ref = float(sigma_ref)
        self.scale = np.sqrt(2)/(len(self.y)**.5*sigma_ref)
        self.offset = op(np.full((op.n, op.n), self.lo))
        self.b = self.scale*(self.y-self.offset)
        # Without an explicit dataset task ID, warm starts are instance-local.
        # Equal noiseless values alone cannot identify a scene/noise repeat.
        identity = task_id or f"instance:{id(self)}"
        self.key = hashlib.sha256((op.identity+str(bounds)+str(sigma_ref)+identity).encode()
                                  + self.y.tobytes()).hexdigest()
        self.calls = {"forward": 0, "adjoint": 0}
        self.limit = None

    def _count(self, which):
        if self.limit is not None and sum(self.calls.values()) >= self.limit:
            raise CallLimit()
        self.calls[which] += 1

    def a(self, z):
        self._count("forward")
        return self.scale*self.delta*self.op(z)

    def at(self, p):
        self._count("adjoint")
        return self.scale*self.delta*self.op.adjoint(p)

    def norm2_bound(self):
        return (self.scale*self.delta)**2*self.op.norm_squared_bound()

    def temperature(self, z):
        return self.lo+self.delta*z


@dataclass
class SolveResult:
    z: np.ndarray
    state: dict
    converged: bool
    status: str
    relative_gap: float
    primal: float
    lower: float
    iterations: int
    calls: dict
    seconds: float
    trace: list = field(default_factory=list)
    checkpoints: dict = field(default_factory=dict)


def _relative(p, lower):
    if lower > p + 1e-8*max(1, abs(p), abs(lower)):
        raise ArithmeticError("invalid lower-bound certificate")
    return max(0., p-lower)/max(1., abs(p), abs(lower))


def _quadratic_lower(primal, grad, z, ridge_curvature):
    # The data Hessian is PSD. Keeping the known ridge curvature strengthens
    # the feasible first-order convex lower bound without estimating a rank.
    s = np.clip(z-grad/ridge_curvature, 0, 1) if ridge_curvature > 0 else np.where(grad >= 0, 0., 1.)
    d = s-z
    return primal+np.sum(grad*d)+.5*ridge_curvature*np.sum(d*d)


def solve(problem, method, lam, tolerance=1e-5, max_calls=10000,
          state=None, checkpoints=(1e-3, 1e-4, 1e-5), check_every=10):
    if method not in ("tikhonov", "tv", "l1") or not np.isfinite(lam) or lam < 0:
        raise ValueError("invalid method or lambda")
    if not 0 < tolerance < 1 or max_calls < 2 or check_every < 1:
        raise ValueError("invalid solver limits")
    if state is not None and (state["key"] != problem.key or state["method"] != method):
        raise ValueError("warm state belongs to a different observation or method")
    shape = (problem.op.n,)*2
    z = np.full(shape, .5) if state is None else np.clip(state["z"].copy(), 0, 1)
    size = z.size
    begin = dict(problem.calls)
    # Reserve the two evaluations needed to certify the final feasible image
    # on a budget exit. These are part of, not additional to, the call cap.
    problem.limit = sum(begin.values())+max_calls-FINAL_CERTIFICATE_CALLS
    # Offset was evaluated at construction; account for it in end-to-end reports.
    start = time.perf_counter()
    trace, saved = [], {}
    pval, lower, gap = float("inf"), -float("inf"), float("inf")
    iteration = 0
    last_change = float("inf")
    dual = np.zeros_like(problem.y) if state is None else state.get("data_dual", np.zeros_like(problem.y)).copy()
    tvdual = np.zeros((2, *shape)) if state is None else state.get("tv_dual", np.zeros((2, *shape))).copy()
    radius = lam/size
    tvdual /= np.maximum(1., np.linalg.norm(tvdual, axis=0)/max(radius, 1e-300))
    if radius == 0:
        tvdual.fill(0)
    norm = np.sqrt(problem.norm2_bound() + (8 if method == "tv" else 0))
    tau = dual_step = .99/max(norm, 1e-15)
    zbar = z.copy()
    status = "call_limit"

    def record(p, l, it):
        rel = _relative(p, l)
        trace.append({"iteration": it, "primal": float(p), "lower": float(l),
                      "relative_gap": rel, "calls": sum(problem.calls.values())-sum(begin.values()),
                      "image_step_rmse_k": last_change, "seconds": time.perf_counter()-start})
        for tol in checkpoints:
            if rel <= tol and str(tol) not in saved:
                saved[str(tol)] = {"z": z.copy(), **trace[-1]}
        return rel

    try:
        if method == "tikhonov":
            diagonal = 2*problem.delta**2*problem.op.w**2/problem.sigma_ref**2+radius
            def hess(d):
                return problem.at(problem.a(d))+radius*d
            while True:
                residual = problem.a(z)-problem.b
                grad = problem.at(residual)+radius*(z-.5)
                pval = .5*np.vdot(residual, residual).real+.5*radius*np.sum((z-.5)**2)
                lower = _quadratic_lower(pval, grad, z, radius)
                gap = record(pval, lower, iteration)
                if gap <= tolerance:
                    status = "converged"
                    break
                old = z.copy()
                # Projected-gradient arc identifies binding variables. Exact
                # quadratic minimization along the feasible chord ensures descent.
                direction = np.clip(z-grad/np.maximum(diagonal, 1e-30), 0, 1)-z
                hd = hess(direction)
                curvature = np.sum(direction*hd)
                alpha = min(1., max(0., -np.sum(grad*direction)/max(curvature, 1e-300)))
                z = np.clip(z+alpha*direction, 0, 1)
                grad = grad+alpha*hd
                free = ~(((z <= 1e-12) & (grad >= 0)) | ((z >= 1-1e-12) & (grad <= 0)))
                # Preconditioned CG in the current free subspace. A boundary
                # hit truncates the step and returns to active-set identification.
                r = -grad*free
                pre = r/np.maximum(diagonal, 1e-30)
                d = pre.copy()
                rho = np.sum(r*pre)
                initial = max(rho, 1e-300)
                for _ in range(min(size, 50)):
                    if rho <= initial*1e-8 or rho <= 1e-28:
                        break
                    hd = hess(d)*free
                    denom = np.sum(d*hd)
                    if denom <= 0:
                        break
                    step = rho/denom
                    bounds = np.full(shape, np.inf)
                    np.divide(1-z, d, out=bounds, where=d > 0)
                    np.divide(-z, d, out=bounds, where=d < 0)
                    cap = max(0., float(bounds.min()))
                    actual = min(step, cap)
                    z = np.clip(z+actual*d, 0, 1)
                    if cap <= step:
                        break
                    r -= step*hd
                    pre = r/np.maximum(diagonal, 1e-30)
                    newrho = np.sum(r*pre)
                    d = pre+(newrho/max(rho, 1e-300))*d
                    rho = newrho
                last_change = problem.delta*float(np.sqrt(np.mean((z-old)**2)))
                iteration += 1
        else:
            while True:
                if iteration % check_every == 0:
                    residual = problem.a(z)-problem.b
                    penalty = (np.linalg.norm(gradient(z), axis=0).sum() if method == "tv" else z.sum())*radius
                    pval = .5*np.vdot(residual, residual).real+penalty
                    s = -problem.at(dual)
                    if method == "tv":
                        s -= gradient_adjoint(tvdual)
                    else:
                        s -= radius
                    lower = -(.5*np.vdot(dual, dual).real+np.vdot(problem.b, dual).real)-np.maximum(s, 0).sum()
                    gap = record(pval, lower, iteration)
                    if gap <= tolerance:
                        status = "converged"
                        break
                dual = (dual+dual_step*(problem.a(zbar)-problem.b))/(1+dual_step)
                if method == "tv":
                    tvdual += dual_step*gradient(zbar)
                    tvdual /= np.maximum(1., np.linalg.norm(tvdual, axis=0)/max(radius, 1e-300))
                    if radius == 0:
                        tvdual.fill(0)
                g = problem.at(dual)
                g += gradient_adjoint(tvdual) if method == "tv" else radius
                newz = np.clip(z-tau*g, 0, 1)
                last_change = problem.delta*float(np.sqrt(np.mean((newz-z)**2)))
                zbar = 2*newz-z
                z = newz
                iteration += 1
    except CallLimit:
        pass
    finally:
        problem.limit = None
    # A limit hit can leave a certificate for an earlier iterate. Never attach
    # that certificate to the returned image or mistake it for convergence.
    if status != "converged":
        residual = problem.a(z)-problem.b
        pval = .5*np.vdot(residual, residual).real
        if method == "tikhonov":
            grad = problem.at(residual)+radius*(z-.5)
            pval += .5*radius*np.sum((z-.5)**2)
            lower = _quadratic_lower(pval, grad, z, radius)
        else:
            pval += radius*(np.linalg.norm(gradient(z), axis=0).sum() if method == "tv" else z.sum())
            s = -problem.at(dual)-(gradient_adjoint(tvdual) if method == "tv" else radius)
            lower = -(.5*np.vdot(dual, dual).real+np.vdot(problem.b, dual).real)-np.maximum(s, 0).sum()
        gap = record(pval, lower, iteration)
    warm = {"key": problem.key, "method": method, "z": z.copy(),
            "data_dual": dual.copy(), "tv_dual": tvdual.copy()}
    return SolveResult(z.copy(), warm, status == "converged", status, gap, float(pval), float(lower),
                       iteration, {k: problem.calls[k]-begin[k] for k in begin},
                       time.perf_counter()-start, trace, saved)


def solve_path(problem, method, lambdas, **options):
    """One observation only, descending lambda, fresh CG for every solve."""
    state, out = None, []
    for lam in sorted(set(map(float, lambdas)), reverse=True):
        result = solve(problem, method, lam, state=state, **options)
        state = result.state
        out.append((lam, result))
    return out
