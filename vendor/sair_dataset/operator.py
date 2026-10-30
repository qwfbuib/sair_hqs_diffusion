"""Separable nonuniform Fourier sampling; no dense measurement matrix."""

import numpy as np


def pixel_weights(n, theta):
    if n < 2 or not np.isfinite(theta) or theta <= 0:
        raise ValueError("n >= 2 and finite theta > 0 are required")
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * (theta / n)
    xi, eta = np.meshgrid(axis, axis, indexing="ij")
    factor = 1 - xi**2 - eta**2
    if np.any(factor <= 0):
        raise ValueError("field of view extends outside the valid direction-cosine disk")
    weights = (theta / n)**2 / np.sqrt(factor)
    return weights / weights.sum()


class ForwardOperator:
    """Real K images -> complex64 K visibilities.

    Exactly repeated u/v coordinates share phase rows; no coordinate rounding
    or merging of measured rows occurs. Each scene uses the same two matrix
    products regardless of batch size, ensuring bitwise batch invariance.
    Loops are over scenes, never over baselines. Workspace is O(N²+UN+VN+UV).
    """

    def __init__(self, n, theta, u, v, w=None):
        self.n = int(n)
        self.theta = float(theta)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if u.ndim != 1 or u.shape != v.shape or not u.size:
            raise ValueError("u/v must be nonempty matching one-dimensional vectors")
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("nonfinite baseline coordinate")
        expected = pixel_weights(self.n, self.theta)
        self.w = expected if w is None else np.asarray(w, dtype=np.float64)
        if self.w.shape != (n, n) or not np.isfinite(self.w).all():
            raise ValueError("invalid pixel weights")
        unique_u, self.iu = np.unique(u, return_inverse=True)
        unique_v, self.iv = np.unique(v, return_inverse=True)
        axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * (theta / n)
        self.b_xi = np.exp(-2j * np.pi * np.outer(unique_u, axis))
        self.a_eta = np.exp(-2j * np.pi * np.outer(unique_v, axis))

    def __call__(self, images):
        x = np.asarray(images, dtype=np.float64)
        single = x.ndim == 2
        if single:
            x = x[None]
        if x.ndim != 3 or x.shape[1:] != (self.n, self.n) or not np.isfinite(x).all():
            raise ValueError(f"expected finite images with shape (...,{self.n},{self.n})")
        result = np.empty((len(x), len(self.iu)), dtype=np.complex64)
        for scene, image in enumerate(x):
            unique_grid = (self.b_xi @ (image * self.w)) @ self.a_eta.T
            result[scene] = unique_grid[self.iu, self.iv]
        return result[0] if single else result


def apply_g(x, n, theta, u, v, w=None):
    """Convenience interface; cache ForwardOperator for repeated use."""
    return ForwardOperator(n, theta, u, v, w)(x)


def decode_gt(u8, bt_min, bt_max):
    return bt_min + (bt_max - bt_min) * np.asarray(u8).astype(np.float64) / 255.0
