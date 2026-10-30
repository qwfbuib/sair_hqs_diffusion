"""NumPy-only map dataset for supervised training on immutable SAIR data.

Compatible with torch.utils.data.DataLoader's map-style protocol without
importing or depending on torch. All four apertures occur equally often in
a complete epoch. Targets and identifiers are outside the model input dict.
"""

from pathlib import Path
import operator
import numpy as np

from .array import gamma_tag
from .reader import DatasetReader, resolve_truth_grid


def measurement_packet(reader, scene, gamma, repeat=0):
    """Read only the selected aperture; pad in original full-baseline order.

    Visibility values and sigma are in K; uv is in wavelengths. No
    normalization, conjugate completion, sorting, or truth feature is added.
    """
    obs = reader.load(scene, gamma, repeat)
    idx = reader.geometry[f"indices_{gamma_tag(gamma)}"]
    y = np.zeros((len(reader.geometry["u"]), 2), dtype=np.float32)
    mask = np.zeros(len(y), dtype=bool)
    y[idx] = np.column_stack((obs["y"].real, obs["y"].imag))
    mask[idx] = True
    return {"visibility_ri": y, "mask": mask,
            "uv": np.column_stack((reader.geometry["u"], reader.geometry["v"])).astype(np.float32),
            "gamma": float(gamma), "sigma": float(obs["sigma"]),
            "physics": {"theta": np.float32(reader.metadata["theta"]),
                        "n_grid": int(reader.metadata["n_grid"]),
                        "bt_min": np.float32(reader.metadata["bt_min"]),
                        "bt_max": np.float32(reader.metadata["bt_max"])}}


class SAIRTrainingDataset:
    """One index = (scene, gamma, noise repeat), in deterministic order.

    `inputs` is the ONLY branch supplied to the model. `target_z` and
    `target_k` are (1,N,N) float32 supervision; target_k is already in K.
    target_size=None uses the measurement's native GT grid (normally 256).
    target_size=128 selects saved gt128 labels when the native grid differs.
    This changes supervision only, never the observation or its physics.
    IDs are for logging/grouping.
    Whole-epoch shuffle samples every configured gamma exactly equally.
    """
    def __init__(self, root, split="train", gammas=None, repeats=None, *, target_size=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        self.root = Path(root).resolve()
        self.reader = DatasetReader(self.root)
        self.target_dir, self.target_n_grid = resolve_truth_grid(
            self.root, self.reader.metadata["n_grid"], target_size)
        self.split = split
        self.scenes = sorted(s for s in self.reader.scenes
                             if self.reader.metadata["class_assignment"][s.split("/")[0]] == split)
        if not self.scenes:
            raise ValueError(f"no selected scenes in split {split}")
        self.gammas = tuple(float(g) for g in (gammas if gammas is not None else
                                             [p["gamma"] for p in self.reader.metadata["plan"]]))
        tags = [gamma_tag(g) for g in self.gammas]
        if not tags or len(tags) != len(set(tags)) or any(t not in self.reader.plan for t in tags):
            raise ValueError("gammas must be unique and present in the dataset")
        self.repeats = tuple(operator.index(r) for r in (repeats if repeats is not None else
                                                       range(self.reader.metadata["noise_repeats"])))
        if not self.repeats or len(self.repeats) != len(set(self.repeats)) or any(
                r < 0 or r >= self.reader.metadata["noise_repeats"] for r in self.repeats):
            raise ValueError("repeats must be unique valid noise repeat indices")
        self.scene_indices = {row["scene"]: row["scene_index"]
                              for row in self.reader.metadata["source_inventory"]}

    def __len__(self):
        return len(self.scenes)*len(self.gammas)*len(self.repeats)

    def key(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("training sample index out of range")
        scene_pos, offset = divmod(index, len(self.gammas)*len(self.repeats))
        gamma_pos, repeat_pos = divmod(offset, len(self.repeats))
        return self.scenes[scene_pos], self.gammas[gamma_pos], self.repeats[repeat_pos]

    def __getitem__(self, index):
        scene, gamma, repeat = self.key(index)
        inputs = measurement_packet(self.reader, scene, gamma, repeat)
        # Read the selected saved grid directly; never resize or fall back to another GT.
        # Only the native GT grid was used to generate the existing observations.
        path = self.target_dir / (scene+".npy")
        u8 = np.load(path, allow_pickle=False)
        n = self.target_n_grid
        if u8.dtype != np.uint8 or u8.shape != (n, n):
            raise ValueError(f"expected ({n},{n}) uint8 ground truth: {path}")
        target_z = u8.astype(np.float32)[None]/np.float32(255.)
        # Evaluate the same K decoding as the reader before float32 conversion.
        lo, hi = self.reader.metadata["bt_min"], self.reader.metadata["bt_max"]
        target_k = (lo+(hi-lo)*u8.astype(np.float64)/255.).astype(np.float32)[None]
        return {"inputs": inputs, "target_z": target_z, "target_k": target_k,
                "target_n_grid": self.target_n_grid,
                "scene": scene, "class_name": scene.split("/")[0], "split": self.split,
                "scene_index": self.scene_indices[scene], "repeat": repeat, "gamma_tag": gamma_tag(gamma)}
