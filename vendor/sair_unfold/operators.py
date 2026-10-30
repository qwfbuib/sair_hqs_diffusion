"""Real images -> complex measured visibilities. All temperatures are in kelvin."""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class CrossGeometry:
    image_size: int = 256
    antennas_x: int = 30
    antennas_y: int = 30
    wavelength_m: float = 0.003
    spacing_lambda: float = 0.875
    range_m: float = 5.0
    scene_width_m: float = 1.0
    include_amplitude: bool = True
    integration: str = "mean"
    chunk_pixels: int = 4096

    def validate(self):
        if min(self.image_size, self.antennas_x, self.antennas_y, self.chunk_pixels) < 1:
            raise ValueError("Grid, antenna counts and chunk size must be positive")
        if min(self.wavelength_m, self.spacing_lambda, self.range_m, self.scene_width_m) <= 0:
            raise ValueError("Geometry lengths must be positive")
        if self.integration not in {"mean", "area", "sum"}:
            raise ValueError("integration must be mean, area or sum")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class CrossSAIROperator(nn.Module):
    """All X-arm / Y-arm antenna correlations, p-major / q-minor ordering.

    G[p,q,n] = w * R^2/(r_p*r_q) * exp(-j*2*pi*(r_p-r_q)/lambda).
    Exact spherical ranges, unit antenna patterns, no dense M x N allocation.
    Row 0 has positive physical y; columns increase in physical x.
    """
    def __init__(self, geometry: CrossGeometry):
        super().__init__()
        self.geometry = geometry.validate()
        self.image_size = geometry.image_size
        self.measurements = geometry.antennas_x * geometry.antennas_y
        n, g = self.image_size, geometry
        # Double precision phase construction avoids large phase cancellation.
        axis = (torch.arange(n, dtype=torch.float64) + 0.5 - n / 2) * g.scene_width_m / n
        yy, xx = torch.meshgrid(axis.flip(0), axis, indexing="ij")
        x, y = xx.flatten(), yy.flatten()
        spacing = g.spacing_lambda * g.wavelength_m
        xa = (torch.arange(g.antennas_x, dtype=torch.float64) - (g.antennas_x - 1) / 2) * spacing
        ya = (torch.arange(g.antennas_y, dtype=torch.float64) - (g.antennas_y - 1) / 2) * spacing
        rp = ((x[None] - xa[:, None]).square() + y[None].square() + g.range_m**2).sqrt()
        rq = (x[None].square() + (y[None] - ya[:, None]).square() + g.range_m**2).sqrt()
        k = 2 * math.pi / g.wavelength_m
        response_x = torch.exp(-1j * k * (rp - g.range_m))
        response_y = torch.exp(-1j * k * (rq - g.range_m))
        if g.include_amplitude:
            response_x = response_x * g.range_m / rp
            response_y = response_y * g.range_m / rq
        self.register_buffer("response_x", response_x.to(torch.complex64))
        self.register_buffer("response_y", response_y.to(torch.complex64))
        self.weight = {"mean": 1 / n**2, "area": (g.scene_width_m / n)**2,
                       "sum": 1.0}[g.integration]

    def forward(self, temperature):
        if temperature.ndim != 4 or temperature.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("Expected temperature [B,1,H,W] matching geometry")
        flat = temperature.flatten(1).to(self.response_x.dtype)
        output = torch.zeros(flat.shape[0], self.geometry.antennas_x, self.geometry.antennas_y,
                             dtype=flat.dtype, device=flat.device)
        for start in range(0, flat.shape[1], self.geometry.chunk_pixels):
            end = start + self.geometry.chunk_pixels
            x = self.response_x[:, start:end]
            y = self.response_y[:, start:end]
            output = output + (flat[:, None, start:end] * x[None]) @ y.conj().T
        return output.flatten(1) * self.weight

    def adjoint(self, visibility):
        """Real Hilbert-space adjoint: Re(G^H v), [B,1,H,W]."""
        v = visibility.reshape(-1, self.geometry.antennas_x, self.geometry.antennas_y)
        chunks = []
        for start in range(0, self.image_size**2, self.geometry.chunk_pixels):
            end = start + self.geometry.chunk_pixels
            x = self.response_x[:, start:end]
            y = self.response_y[:, start:end]
            chunks.append(torch.einsum("bpq,pn,qn->bn", v, x.conj(), y).real)
        return torch.cat(chunks, dim=1).reshape(-1, 1, self.image_size, self.image_size) * self.weight

    def metadata(self):
        return {"kind": "cross_spherical", "geometry": asdict(self.geometry)}


class MatrixSAIROperator(nn.Module):
    """Adapter for an existing G[M,H*W], without changing its scale or convention."""
    def __init__(self, matrix, image_size, chunk_rows=256):
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[1] != image_size**2 or matrix.shape[0] < 1:
            raise ValueError(f"G must have shape [M,{image_size**2}]")
        if chunk_rows < 1 or not np.isfinite(matrix).all():
            raise ValueError("G must be finite and chunk_rows positive")
        self.register_buffer("matrix", torch.as_tensor(np.asarray(matrix, dtype=np.complex64).copy()))
        self.image_size, self.measurements = image_size, matrix.shape[0]
        self.chunk_rows = chunk_rows

    def forward(self, temperature):
        if temperature.ndim != 4 or temperature.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("Expected temperature [B,1,H,W] matching G")
        flat = temperature.flatten(1).to(self.matrix.dtype)
        return torch.cat([flat @ rows.T for rows in self.matrix.split(self.chunk_rows)], dim=1)

    def adjoint(self, visibility):
        return (visibility @ self.matrix.conj()).real.reshape(-1, 1, self.image_size, self.image_size)

    def metadata(self):
        return {"kind": "matrix", "shape": list(self.matrix.shape),
                "image_size": self.image_size, "ordering": "C-row-major"}


def load_operator(image_size, device, geometry_path=None, matrix_path=None, matrix_key="G"):
    if geometry_path and matrix_path:
        raise ValueError("Choose geometry or G matrix, not both")
    if matrix_path:
        path = Path(matrix_path)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as values:
                if matrix_key not in values:
                    raise ValueError(f"Missing matrix key {matrix_key}; keys: {values.files}")
                matrix = values[matrix_key]
        else:
            matrix = np.load(path, allow_pickle=False)
        operator = MatrixSAIROperator(matrix, image_size)
    else:
        geometry = (CrossGeometry.load(geometry_path) if geometry_path
                    else CrossGeometry(image_size=image_size))
        if geometry.image_size != image_size:
            raise ValueError("Geometry grid and model grid differ; rebuild G on the HR grid")
        operator = CrossSAIROperator(geometry)
    return operator.to(device)


def complex_noise(reference, sigma, generator):
    """sigma^2 = E|n_m|^2; each real/imag component has variance sigma^2/2."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    real = torch.randn(reference.shape, device=reference.device, generator=generator)
    imag = torch.randn(reference.shape, device=reference.device, generator=generator)
    return sigma / math.sqrt(2) * torch.complex(real, imag)


def measurement_loss(predicted, measured, sigma):
    if sigma <= 0:
        raise ValueError("Likelihood sigma must be positive, including for noiseless simulation")
    # Mean per complex measurement; total negative log likelihood is M times this.
    return (predicted - measured).abs().square().mean() / sigma**2
