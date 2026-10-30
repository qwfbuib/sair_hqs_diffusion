"""Frozen scene lists, independent seed families, shared-network boundary."""

import hashlib
import json
from pathlib import Path
import numpy as np
from . import PROTOCOL_VERSION
from sair_dataset.array import gamma_tag
from sair_dataset.training import measurement_packet

SEED = 20261023
FAMILIES = {"selection": 1, "precision": 2, "l1_tuning": 3, "calibration": 4,
            "confirmation": 5, "exploration": 6, "bootstrap": 7, "controls": 8,
            "offset_single": 9}
BASE_TASKS = {"tikhonov": 7588, "tv": 7588, "l1": 17248, "clean": 15328, "network": 22328}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_manifest(reader):
    # Production lists require the complete dataset, never a pilot prefix.
    metadata = reader.metadata
    classes = metadata["class_assignment"]
    selected = sorted(reader.scenes)
    grouped = {c: sorted(s for s in selected if s.split("/")[0] == c) for c in sorted(classes)}
    if sorted(list(classes.values()).count(x) for x in ("train", "val", "test")) != [7, 7, 31]:
        raise ValueError("expected class-exclusive 31/7/7 split")
    if any(len(s) != 700 for s in grouped.values()):
        raise ValueError("manifest requires the complete 700 scenes per class")
    ordered = {}
    for i, (c, scenes) in enumerate(grouped.items()):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, FAMILIES["selection"], i]))
        ordered[c] = [scenes[k] for k in rng.permutation(len(scenes))]
    def take(split, count):
        return [s for c in ordered if classes[c] == split for s in ordered[c][:count]]
    precision = []
    valclasses = [c for c in ordered if classes[c] == "val"]
    for gi, gamma in enumerate((1., .75, .5, .35)):
        for j in range(5):
            c = valclasses[(gi*5+j) % 7]
            precision.append({"scene": ordered[c][30+gi], "gamma": gamma, "repeat": 0})
    out = {"protocol_version": PROTOCOL_VERSION, "seed": SEED, "seed_families": FAMILIES,
           "dataset_metadata_sha256": hashlib.sha256((reader.root/"scenes.json").read_bytes()).hexdigest(),
           "class_assignment": classes, "main": take("test", 200), "robustness": take("test", 100),
           "network_extra": [s for c in ordered if classes[c] == "test" for s in ordered[c][200:]],
           "tuning_coarse": take("val", 4), "tuning_fine": take("val", 10),
           "tuning_final": take("val", 20), "precision_natural": precision,
           "baseline_tasks_before_cache": BASE_TASKS}
    out["manifest_sha256"] = fingerprint(out)
    return out


def network_packet(reader, scene, gamma, repeat=0):
    """Only this aperture is read. Truth and other apertures never enter."""
    return measurement_packet(reader, scene, gamma, repeat)


class SharedNetwork:
    """Adapter around a user-supplied predict(packet) callable, ONE checkpoint."""
    def __init__(self, predict, checkpoint_path):
        self.predict = predict
        self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()

    def __call__(self, packet):
        out = np.asarray(self.predict(packet), float)
        n = packet["physics"]["n_grid"]
        if out.shape != (n, n) or not np.isfinite(out).all():
            raise ValueError("network must return one finite K image on the common grid")
        return out


def choose_checkpoint(validation_records):
    """records: checkpoint, class, gamma, split, rmse. Equal class/gamma mean."""
    groups = {}
    for r in validation_records:
        if r["split"] != "val" or not np.isfinite(r["rmse"]):
            raise ValueError("checkpoint selection requires finite validation-only results")
        groups.setdefault(r["checkpoint"], {}).setdefault((r["class"], r["gamma"]), []).append(r["rmse"])
    expected = None
    scores = {}
    for checkpoint, cells in groups.items():
        if expected is None:
            expected = set(cells)
        if set(cells) != expected or len({c for c, _ in cells}) != 7 or {g for _, g in cells} != {1., .75, .5, .35} or len(cells) != 28:
            raise ValueError("every checkpoint must cover all seven validation classes and four gammas")
        scores[checkpoint] = float(np.mean([np.mean(v) for v in cells.values()]))
    if not scores:
        raise ValueError("no validation records")
    return min(scores, key=lambda k: (scores[k], k)), scores
