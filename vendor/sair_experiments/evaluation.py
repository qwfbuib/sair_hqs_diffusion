"""Metrics, numerical observable-space projection, and exact certification."""

import numpy as np
from scipy.stats import beta, binom


def image_metrics(prediction, truth, dynamic_range=170.):
    e = np.asarray(prediction)-np.asarray(truth)
    if e.ndim != 2 or not np.isfinite(e).all() or dynamic_range <= 0:
        raise ValueError("finite aligned images and positive range required")
    rmse = float(np.sqrt(np.mean(e**2)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()),
            "psnr": float(20*np.log10(dynamic_range/rmse)) if rmse else float("inf")}


def observation_metrics(op, prediction, y, sigma_ref, clean=None):
    estimate = op(prediction)
    out = {"normalized_noisy_residual": float(np.mean(np.abs(estimate-y)**2)/sigma_ref**2)}
    if clean is not None:
        out["clean_visibility_rmse"] = float(np.sqrt(np.mean(np.abs(estimate-clean)**2)))
    return out


class NumericalProjector:
    """Block row Gram, O((2M)^2) storage; no dense production G.

    This is intentionally an explicit precomputation. At 1830 rows it is
    expensive and must be included separately in the wall-clock report.
    """
    def __init__(self, op, cutoff=1e-10, block_pixels=512, gram=None):
        if not 0 < cutoff < 1:
            raise ValueError("relative eigenvalue cutoff must be in (0,1)")
        self.op, self.cutoff = op, float(cutoff)
        m, n = len(op.u), op.n
        if gram is None:
            gram = np.zeros((2*m, 2*m))
            xx, yy = np.meshgrid(op.axis, op.axis, indexing="ij")
            for start in range(0, n*n, block_pixels):
                sl = slice(start, start+block_pixels)
                phase = np.exp(-2j*np.pi*(op.u[:, None]*xx.ravel()[sl]+op.v[:, None]*yy.ravel()[sl]))
                g = phase*op.w.ravel()[sl]
                a = np.concatenate((g.real, g.imag), axis=0)
                gram += a@a.T
        values, vectors = np.linalg.eigh(gram)
        self.gram = gram
        keep = values > max(0., float(values[-1]))*cutoff
        keep &= values > 0
        self.values, self.vectors = values[keep], vectors[:, keep]
        self.rank = len(self.values)

    def project(self, e):
        g = self.op(e)
        ae = np.concatenate((g.real, g.imag))
        coeff = self.vectors@((self.vectors.T@ae)/self.values)
        m = len(g)
        return self.op.adjoint(coeff[:m]+1j*coeff[m:])

    def metrics(self, e):
        observable = self.project(e)
        complement = e-observable
        return {"rmse_range_num": float(np.sqrt(np.mean(observable**2))),
                "rmse_null_num": float(np.sqrt(np.mean(complement**2))),
                "projection_cutoff": self.cutoff, "projection_rank": self.rank,
                "projection_energy_residual": float(np.mean(e**2)-np.mean(observable**2)-np.mean(complement**2))}


def cp_interval(successes, trials, family_size=60, alpha=.05):
    if not 0 <= successes <= trials or trials < 1 or family_size < 1 or not 0 < alpha < 1:
        raise ValueError("invalid binomial counts or confidence family")
    tail = alpha/(2*family_size)
    return (0. if successes == 0 else float(beta.ppf(tail, successes, trials-successes+1)),
            1. if successes == trials else float(beta.ppf(1-tail, successes+1, trials-successes)))


def certification_boundary(n=650, target=.9, true_probability=.95):
    k = next(k for k in range(n+1) if cp_interval(k, n)[0] >= target)
    return {"trials": n, "critical_successes": k,
            "power": float(binom.sf(k-1, n, true_probability)),
            "lower_at_boundary": cp_interval(k, n)[0]}


def certify_separation(counts, fwhm, false_blank, false_single, trials=650):
    """counts is {distance_in_direction_cosines: success_count}, ONE direction.

    The caller must submit the complete predeclared 54+6 family together;
    this function never infers a new family size from available results.
    """
    if trials != 650 or fwhm <= 0:
        raise ValueError("protocol requires 650 repeats and a positive reference FWHM")
    intervals = {float(d): cp_interval(k, trials) for d, k in counts.items()}
    passing = [d for d, ci in intervals.items() if ci[0] >= .9]
    d90 = min(passing) if passing else None
    controls = {"blank": cp_interval(false_blank, trials), "single": cp_interval(false_single, trials)}
    passed = d90 is not None and d90 < fwhm and all(ci[1] <= .05 for ci in controls.values())
    return {"d90_cert": d90, "intervals": intervals, "controls": controls,
            "superresolution_certified": passed, "status": "certified" if passed else "uncertified"}


def paired_bootstrap(differences, classes, repeats=2000, seed=20261023):
    """Resample classes, then aligned scene differences within each class."""
    differences, classes = np.asarray(differences, float), np.asarray(classes)
    if differences.shape != classes.shape or not np.isfinite(differences).all():
        raise ValueError("finite paired differences and class labels required")
    groups = [differences[classes == c] for c in np.unique(classes)]
    if not groups:
        raise ValueError("empty comparison")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    draws = np.empty(repeats)
    for j in range(repeats):
        sampled = rng.integers(len(groups), size=len(groups))
        draws[j] = np.mean([rng.choice(groups[k], len(groups[k]), replace=True).mean() for k in sampled])
    return {"mean_difference": float(np.mean([g.mean() for g in groups])),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "repeats": repeats,
            "per_class": {str(c): float(differences[classes == c].mean()) for c in np.unique(classes)}}


def paired_report(records, left, right, metric="rmse"):
    """Explicit completeness check prevents silent deletion of failed pairs."""
    groups = {}
    for r in records:
        if r.get("method") not in (left, right):
            continue
        key = r["scene"], r["gamma"], r.get("repeat", 0)
        cell = groups.setdefault(key, {})
        if r["method"] in cell:
            raise ValueError("duplicate paired observation")
        cell[r["method"]] = r
    by_gamma = {}
    for gamma in sorted({k[1] for k in groups}):
        subset = {k: v for k, v in groups.items() if k[1] == gamma}
        failures = [k for k, v in subset.items() if set(v) != {left, right} or any(
            metric not in r or not r.get("converged", False) or not np.isfinite(r.get(metric, np.nan)) for r in v.values())]
        record = {"pairs_expected": len(subset), "failed_or_missing_pairs": failures,
                  "failure_rate": len(failures)/len(subset), "paired_conclusion_available": not failures}
        if not failures:
            record.update(paired_bootstrap([v[left][metric]-v[right][metric] for v in subset.values()],
                                          [k[0].split("/")[0] for k in subset]))
        by_gamma[str(gamma)] = record
    return {"left_minus_right": [left, right], "metric": metric, "by_gamma": by_gamma,
            "note": "failed pairs are listed; no complete-case-only victory conclusion"}
