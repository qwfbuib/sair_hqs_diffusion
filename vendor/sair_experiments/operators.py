"""Float64/complex128 forward, real adjoint, and certified norm bounds."""

import hashlib
from pathlib import Path
import numpy as np

from sair_dataset.array import all_baselines, y_array, gamma_tag
from sair_dataset.operator import ForwardOperator


class VisibilityOperator(ForwardOperator):
    def __init__(self, n, theta, u, v):
        super().__init__(n, theta, u, v)
        self.u, self.v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        self.axis = (np.arange(n) - (n - 1) / 2) * theta / n
        digest = hashlib.sha256()
        for a in (self.u, self.v, self.w):
            digest.update(a.tobytes())
        self.identity = digest.hexdigest()
        self._bound = None

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n, self.n) or not np.isfinite(x).all():
            raise ValueError("expected one finite image on the operator grid")
        grid = (self.b_xi @ (x * self.w)) @ self.a_eta.T
        return grid[self.iu, self.iv]

    def adjoint(self, y):
        """Adjoint for real images: Re(G^H y), with repeated rows retained."""
        y = np.asarray(y, dtype=np.complex128)
        if y.shape != self.u.shape or not np.isfinite(y).all():
            raise ValueError("invalid visibility vector")
        grid = np.zeros((len(self.b_xi), len(self.a_eta)), dtype=complex)
        np.add.at(grid, (self.iu, self.iv), y)
        return (self.b_xi.conj().T @ grid @ self.a_eta.conj()).real * self.w

    def reference(self, x, block=16):
        """Independent direct pixel sum, for audits, not reconstruction."""
        out = np.empty(len(self.u), complex)
        for start in range(0, len(out), block):
            u, v = self.u[start:start+block], self.v[start:start+block]
            phase = np.exp(-2j*np.pi*(u[:, None, None]*self.axis[None, :, None]
                                      + v[:, None, None]*self.axis[None, None, :]))
            out[start:start+block] = np.sum(phase * (np.asarray(x)*self.w), axis=(1, 2))
        return out

    def norm_squared_bound(self):
        """Certified bound, not a power-iteration estimate.

        G=F diag(w). The absolute row sum of FF^H bounds its spectral
        radius; FF^H is a product of two Dirichlet kernels. Real-domain
        restriction cannot increase the norm. No conjugate rows are added.
        """
        if self._bound is None:
            n, step = self.n, self.theta/self.n
            def kernel(d):
                t = d*step
                # Centered finite Fourier sum; sinc handles the zero limit.
                nearest = np.rint(t)
                q = t-nearest
                return n*np.abs(np.sinc(n*q)/np.sinc(q))
            bound = 0.0
            for start in range(0, len(self.u), 128):
                du = self.u[start:start+128, None]-self.u[None]
                dv = self.v[start:start+128, None]-self.v[None]
                bound = max(bound, float(np.max(np.sum(kernel(du)*kernel(dv), axis=1))))
            self._bound = bound * float(self.w.max())**2 * (1+1e-12)
        return self._bound

    def readonly_cache(self, directory):
        """Share immutable numeric arrays between spawned CPU workers."""
        root = Path(directory)/self.identity
        root.mkdir(parents=True, exist_ok=True)
        for name in ("u", "v", "w", "b_xi", "a_eta", "iu", "iv", "axis"):
            path = root/(name+".npy")
            value = getattr(self, name)
            if not path.exists():
                with path.open("xb") as stream:
                    np.save(stream, value)
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            if saved.shape != value.shape or saved.dtype != value.dtype or not np.array_equal(saved, value):
                raise ValueError("immutable geometry cache mismatch")
            setattr(self, name, saved)
        return self


def dataset_operator(reader, gamma, cache=None):
    c = reader.metadata["config"]
    u, v = all_baselines(y_array(c["n_arm"], c["spacing_lambda"], c["freq"]), c["freq"])
    if not (np.array_equal(u.astype("float32"), reader.geometry["u"])
            and np.array_equal(v.astype("float32"), reader.geometry["v"])):
        raise ValueError("saved geometry disagrees with float64 array definition")
    idx = reader.geometry[f"indices_{gamma_tag(gamma)}"]
    operator = VisibilityOperator(c["n_grid"], c["theta"], u[idx], v[idx])
    return operator.readonly_cache(cache) if cache is not None else operator


def gradient(x):
    g = np.zeros((2, *x.shape))
    g[0, :-1] = x[1:]-x[:-1]
    g[1, :, :-1] = x[:, 1:]-x[:, :-1]
    return g


def gradient_adjoint(p):
    out = np.zeros(p.shape[1:])
    out[:-1] -= p[0, :-1]
    out[1:] += p[0, :-1]
    out[:, :-1] -= p[1, :, :-1]
    out[:, 1:] += p[1, :, :-1]
    return out
