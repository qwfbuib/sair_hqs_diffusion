"""Read one observation from the scene-grouped shared-geometry format."""

import json
import operator
from pathlib import Path

import numpy as np

from .array import gamma_tag
from .generate import SCHEMA_VERSION
from .operator import decode_gt


def resolve_truth_grid(root, n_grid, target_size=None):
    """Select an existing saved GT grid without resizing or fallback."""
    native_n = int(n_grid)
    try:
        n = native_n if target_size is None else operator.index(target_size)
    except TypeError as exc:
        raise ValueError("target_size must be None or an integer grid size") from exc
    if n not in (native_n, 128):
        raise ValueError(f"target_size must be None, {native_n}, or 128")
    directory = Path(root) / ("gt" if n == native_n else "gt128")
    if not directory.is_dir():
        raise FileNotFoundError(f"target GT directory does not exist: {directory}")
    return directory, n


class DatasetReader:
    """Cache metadata and shared geometry once when loading many scenes."""
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "scenes.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != SCHEMA_VERSION or self.metadata.get("status") != "complete":
            raise ValueError("unsupported or incomplete dataset")
        with np.load(self.root / "geometry.npz", allow_pickle=False) as archive:
            self.geometry = {key: archive[key] for key in archive.files}
        self.scenes = set(self.metadata["selected_scenes"])
        self.plan = {p["tag"]: p for p in self.metadata["plan"]}

    def load(self, scene, gamma=1.0, repeat=0):
        scene = str(scene).replace("\\", "/")
        if scene not in self.scenes:
            raise KeyError(f"unknown scene ID (expected class/stem): {scene}")
        tag = gamma_tag(gamma)
        if tag not in self.plan:
            raise KeyError(f"gamma is absent from dataset: {gamma}")
        if not isinstance(repeat, (int, np.integer)) or not 0 <= repeat < self.metadata["noise_repeats"]:
            raise IndexError("noise repeat out of range")
        indices = self.geometry[f"indices_{tag}"]
        with np.load(self.root / "meas" / (scene + ".npz"), allow_pickle=False) as archive:
            result = {"y": archive[f"y_{tag}"][repeat].copy(),
                      "sigma": archive[f"sigma_{tag}"][repeat],
                      "snr_db": archive[f"snr_db_{tag}"][repeat]}
        result.update(self.plan[tag])
        result.update(u=self.geometry["u"][indices].copy(), v=self.geometry["v"][indices].copy(),
                      theta=self.metadata["theta"], n_grid=self.metadata["n_grid"],
                      scene=scene, repeat=int(repeat),
                      split=self.metadata["class_assignment"][scene.split("/")[0]])
        return result

    def truth(self, scene, *, target_size=None):
        """Return an (N,N) float64 K image; default native GT, or saved 128 GT."""
        scene = str(scene).replace("\\", "/")
        if scene not in self.scenes:
            raise KeyError(scene)
        directory, n = resolve_truth_grid(self.root, self.metadata["n_grid"], target_size)
        path = directory / (scene + ".npy")
        u8 = np.load(path, allow_pickle=False)
        if u8.dtype != np.uint8 or u8.shape != (n, n):
            raise ValueError(f"expected ({n},{n}) uint8 ground truth: {path}")
        return decode_gt(u8, self.metadata["bt_min"], self.metadata["bt_max"])


def load_measurement(root, scene, gamma=1.0, repeat=0):
    return DatasetReader(root).load(scene, gamma, repeat)
