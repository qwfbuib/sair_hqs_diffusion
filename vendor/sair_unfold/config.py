from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math

@dataclass
class Config:
    image_size: int = 256
    t_min: float = 100.0
    t_max: float = 300.0
    base_channels: int = 32
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    stages: int = 8
    cg_iters: int = 20
    cg_rtol: float = 1e-5
    initial_beta: float = 0.05
    beta_start: float = 0.1
    beta_end: float = 0.01
    mu_start: float = 0.2
    mu_end: float = 0.01
    sigma_min: float = 0.01
    sigma_max: float = 0.5
    sigma_start: float = 0.25
    sigma_end: float = 0.02
    alpha_init: float = 0.5
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 0
    learning_rate: float = 2e-4
    noise_sigma: float = 0.5
    visibility_weight: float = 0.001
    gradient_weight: float = 0.05
    range_weight: float = 0.01
    dsm_weight: float = 0.01

    def validate(self):
        for name in ("image_size", "base_channels", "stages", "cg_iters", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.base_channels < 4 or self.image_size < 8 or not self.channel_mult
                or any(not isinstance(m, int) or m < 1 for m in self.channel_mult)
                or self.image_size % 2**(len(self.channel_mult)-1)):
            raise ValueError("Invalid U-Net dimensions")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("Invalid num_workers")
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.t_min >= self.t_max:
            raise ValueError("t_min must be below t_max")
        for key in ("cg_rtol", "initial_beta", "beta_start", "beta_end", "mu_start",
                    "mu_end", "sigma_min", "sigma_max", "learning_rate", "noise_sigma"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not self.sigma_min < self.sigma_end <= self.sigma_start < self.sigma_max:
            raise ValueError("Require sigma_min < sigma_end <= sigma_start < sigma_max")
        if not 0 < self.alpha_init < 1:
            raise ValueError("alpha_init must be in (0,1)")
        for key in ("visibility_weight", "gradient_weight", "range_weight", "dsm_weight"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        return self

    @property
    def scale(self):
        return (self.t_max - self.t_min) / 2

    @property
    def center(self):
        return (self.t_max + self.t_min) / 2

    def normalize(self, temperature):
        return (temperature - self.center) / self.scale

    def denormalize(self, x):
        return x * self.scale + self.center

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig"))).validate()
