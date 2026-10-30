"""Point workload and wall-clock workload, with counts independently derivable."""

from .points import dirty_fwhm
from .design import FAMILIES


def point_plan(op):
    fwhm = {str(a): dirty_fwhm(op, a) for a in (0, 15, 30)}
    configs = []
    for center in ((0., 0.), (-op.theta/8, 0.), (op.theta/8, op.theta/8)):
        for angle in (0, 15, 30):
            for ratio in (1., 2.):
                for factor in (.4, .6, .8, 1., 1.25, 1.5):
                    primary = center == (0., 0.) and ratio == 1
                    configs.append({"config_id": len(configs), "center": center, "degrees": angle,
                                    "ratio": ratio, "distance": fwhm[str(angle)]*factor,
                                    "fwhm": fwhm[str(angle)], "distance_factor": factor,
                                    "sources": 2, "noise_repeats": 650 if primary else 10,
                                    "family": FAMILIES["confirmation" if primary else "exploration"],
                                    "confirmatory": primary, "include_noiseless": True})
    return {"directional_fwhm": fwhm, "configurations": configs,
            "calibration": {"blank": 650, "center_single": 650, "family": FAMILIES["calibration"]},
            "controls": {"blank": 650, "center_single": 650, "family": FAMILIES["controls"]},
            "offset_single": {"count": 20, "family": FAMILIES["offset_single"]},
            "tuning": {"double": 60, "single": 30, "blank": 30, "family": FAMILIES["l1_tuning"]},
            "formal_observations_per_reconstruction_method": sum(c["noise_repeats"]+1 for c in configs)+1320}


def base_workload(tolerance=1e-5):
    """Counts in paths for tuning (not mislabeled independent cold solves)."""
    rows = []
    def add(method, gamma, phase, scene_type, count, tol=tolerance, batch=1, lambdas_per_path=None):
        row = dict(method=method, gamma=gamma, phase=phase, scene_type=scene_type,
                   tolerance=tol, batch=batch, count=count, unit="tasks")
        if lambdas_per_path:
            row.update(unit="whole_lambda_paths", lambdas_per_path=lambdas_per_path)
        rows.append(row)
    for method in ("tikhonov", "tv"):
        for gamma in (1., .75, .5, .35):
            add(method, gamma, "main" if gamma == 1 else "robustness", "natural", 1400 if gamma == 1 else 700)
            for phase, count, length, tol in (("coarse", 28, 9, 1e-3), ("fine", 70, 7, tolerance), ("final", 140, 2, tolerance)):
                add(method, gamma, phase, "natural", count, tol, lambdas_per_path=length)
    for gamma in (1., .75, .5, .35):
        add("network", gamma, "main" if gamma == 1 else "robustness", "natural", 1400 if gamma == 1 else 700, None)
    add("network", 1., "extra", "natural", 3500, None)
    for method in ("l1", "clean", "network"):
        add(method, 1., "point_formal", "synthetic", 14028, tolerance if method == "l1" else None)
        add(method, 1., "calibration", "synthetic", 1300, tolerance if method == "l1" else None)
    add("l1", 1., "coarse", "synthetic", 120, 1e-3, lambdas_per_path=9)
    add("l1", 1., "fine", "synthetic", 120, tolerance, lambdas_per_path=7)
    return rows
