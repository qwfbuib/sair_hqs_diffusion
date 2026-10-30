from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class TemperatureDataset(Dataset):
    """Memory-mapped, absolute kelvin arrays [N,1,H,W]."""
    def __init__(self, path, config):
        self.path, self.config = str(path), config
        self.array = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.array.ndim != 4 or self.array.shape[1:] != (1, config.image_size, config.image_size):
            raise ValueError("Dataset must have shape [N,1,H,W] matching model config")
        if len(self.array) == 0:
            raise ValueError("Empty dataset split")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        temperature = torch.from_numpy(np.array(self.array[index], dtype=np.float32, copy=True))
        if not torch.isfinite(temperature).all():
            raise ValueError(f"Non-finite temperature in {self.path} sample {index}")
        if temperature.min() < self.config.t_min - 1e-3 or temperature.max() > self.config.t_max + 1e-3:
            raise ValueError("Temperature exceeds configured range; check input units")
        return self.config.normalize(temperature)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["array"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)


def check_dataset_metadata(data_dir, config):
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise ValueError("Missing manifest.json; prepare the dataset first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("image_size", "t_min", "t_max"):
        if manifest[key] != getattr(config, key):
            raise ValueError(f"Dataset {key} differs from config")
    return manifest


def image_temperature(path, config, allow_resize=False):
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (config.image_size, config.image_size):
            if not allow_resize:
                raise ValueError(f"{path}: expected {config.image_size}x{config.image_size}, "
                                 f"got {image.size}; use --allow-resize explicitly")
            image = image.resize((config.image_size, config.image_size), Image.Resampling.BOX)
        gray = np.asarray(image, dtype=np.float32) / 255.0
    return (config.t_min + gray * (config.t_max - config.t_min))[None]
