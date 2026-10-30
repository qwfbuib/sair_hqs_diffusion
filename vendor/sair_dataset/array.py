"""Y-array geometry and conjugate-aware baseline accounting."""

from dataclasses import dataclass

import numpy as np

C_M_S = 299_792_458.0
APERTURE_ATOL = 1e-9


def y_array(n_per_arm=21, spacing_lambda=0.875, freq=1.4135e9):
    """Return (antenna, xy) positions in metres, float64, hub once."""
    if n_per_arm < 2 or not np.isfinite(spacing_lambda) or spacing_lambda <= 0:
        raise ValueError("n_per_arm >= 2 and spacing_lambda > 0 are required")
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError("freq must be positive and finite")
    distance = np.arange(1, n_per_arm, dtype=np.float64) * spacing_lambda * C_M_S / freq
    angles = np.deg2rad(np.arange(3, dtype=np.float64) * 120.0)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    arms = distance[None, :, None] * directions[:, None, :]
    return np.concatenate((np.zeros((1, 2)), arms.reshape(-1, 2)))


def all_baselines(positions, freq=1.4135e9):
    positions = np.asarray(positions, dtype=np.float64)
    i, j = np.triu_indices(len(positions), k=1)
    uv = (positions[i] - positions[j]) / (C_M_S / freq)
    return uv[:, 0], uv[:, 1]


def count_independent(u, v, decimals=6):
    """Merge repeated and opposite rows, never an isolated y reflection.

    Rounding precedes sign canonicalization, so roundoff near the v axis
    cannot split an opposite pair. It is used ONLY for this count.
    """
    q = np.round(np.column_stack((u, v)).astype(np.float64), decimals)
    flip = (q[:, 0] < 0) | ((q[:, 0] == 0) & (q[:, 1] < 0))
    q = np.where(flip[:, None], -q, q)
    return len(np.unique(q, axis=0))


def gamma_tag(gamma):
    value = float(gamma)
    percent = round(value * 100)
    if not np.isfinite(value) or not 0 < value <= 1 or abs(value * 100 - percent) > 1e-8:
        raise ValueError("gammas must be distinct integer percentages in (0, 1]")
    return f"g{percent:03d}"


@dataclass(frozen=True)
class Aperture:
    gamma: float
    tag: str
    indices: np.ndarray
    m: int
    n_independent: int
    n_real_constraints_upper: int
    cells: float
    null_fraction: float
    r_max_actual: float

    def metadata(self):
        return {key: getattr(self, key) for key in (
            "gamma", "tag", "m", "n_independent", "n_real_constraints_upper",
            "cells", "null_fraction", "r_max_actual")}


def aperture_plan(u, v, gammas, theta, n_grid):
    r = np.hypot(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    r_max = float(r.max())
    if r_max <= 0:
        raise ValueError("array must contain nonzero baselines")
    tags = [gamma_tag(g) for g in gammas]
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("gammas must be nonempty and unique")
    plan = []
    for gamma, tag in zip(gammas, tags):
        indices = np.flatnonzero(r <= gamma * r_max + APERTURE_ATOL).astype(np.int32)
        if not len(indices):
            raise ValueError(f"empty aperture: {tag}")
        independent = count_independent(u[indices], v[indices])
        real_upper = 2 * independent
        plan.append(Aperture(float(gamma), tag, indices, len(indices), independent,
                             real_upper, float(2 * gamma * r_max * theta),
                             float(1 - min(real_upper, n_grid**2) / n_grid**2),
                             float(r[indices].max())))
    return plan
