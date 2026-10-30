"""Generate scene-grouped SAIR measurements directly from the truth grid."""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .array import APERTURE_ATOL, all_baselines, aperture_plan, y_array
from .operator import ForwardOperator, decode_gt, pixel_weights

SCHEMA_VERSION = "sair-scene-v2"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUMMARY_FIELDS = ["scene", "file", "class_name", "split", "scene_index", "gamma",
                  "tag", "repeat", "m", "n_independent", "n_real_constraints_upper",
                  "cells", "null_fraction", "sigma", "median_abs_v", "snr_db"]


@dataclass(frozen=True)
class Config:
    input: Path
    output: Path
    theta: float = 1.05
    n_grid: int = 256
    gammas: tuple = (1.0, 0.75, 0.5, 0.35)
    n_arm: int = 21
    spacing_lambda: float = 0.875
    freq: float = 1.4135e9
    tsys: float = 350.0
    bandwidth: float = 19e6
    tau: float = 1.2
    noise_scale: float = 1.0
    snr_db: float | None = None
    noise_repeats: int = 1
    per_class: int = 40
    seed: int = 20261022
    batch: int = 8
    bt_min: float = 150.0
    bt_max: float = 320.0

    def validate(self):
        if self.per_class < 0 or self.noise_repeats < 1 or self.batch < 1 or self.seed < 0:
            raise ValueError("invalid scene count, repeats, batch, or seed")
        for name in ("tsys", "bandwidth", "tau"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.noise_scale) or self.noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if self.snr_db is not None and not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        if not 0 <= self.bt_min < self.bt_max or not math.isfinite(self.bt_max):
            raise ValueError("require finite 0 <= bt_min < bt_max")
        source, output = Path(self.input).resolve(), Path(self.output).resolve()
        if not source.is_dir():
            raise ValueError(f"input directory does not exist: {source}")
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("input and output directories must be separate trees")
        pixel_weights(self.n_grid, self.theta)


def collect_inventory(root):
    """Index the ENTIRE source tree before applying per-class selection."""
    root = Path(root)
    inventory = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in directory.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            continue
        seen = set()
        for path in files:
            if path.stem in seen:
                raise ValueError(f"duplicate scene stem in class {directory.name}: {path.stem}")
            seen.add(path.stem)
            inventory.append({"scene": f"{directory.name}/{path.stem}",
                              "source": path.relative_to(root).as_posix(),
                              "class_name": directory.name, "scene_index": len(inventory)})
    if not inventory:
        raise ValueError("no images in class subdirectories")
    return inventory


def select_inventory(inventory, per_class):
    counts = {}
    selected = []
    for item in inventory:
        cls = item["class_name"]
        counts[cls] = counts.get(cls, 0) + 1
        if per_class == 0 or counts[cls] <= per_class:
            selected.append(item)
    return selected


def split_classes(classes, seed):
    """Largest-remainder 70/15/15 allocation; 45 classes -> 31/7/7."""
    classes = sorted(classes)
    exact = np.array([0.70, 0.15, 0.15]) * len(classes)
    sizes = np.floor(exact).astype(int)
    order = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))
    for i in order[:len(classes) - int(sizes.sum())]:
        sizes[i] += 1
    shuffled = np.random.default_rng(seed).permutation(classes)
    labels = np.repeat(["train", "val", "test"], sizes)
    return dict(sorted(zip(shuffled.tolist(), labels.tolist())))


def read_u8(path, n):
    with Image.open(path) as image:
        if image.size != (n, n):
            raise ValueError(f"wrong image dimensions {image.size}, expected {(n,n)}: {path}")
        return np.array(image.convert("L"), dtype=np.uint8)


def sigma_for(clean, config):
    median = float(np.median(np.abs(clean)))
    if config.snr_db is not None:
        sigma = median * 10 ** (-config.snr_db / 20)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("SNR override produced a nonpositive/nonfinite noise scale")
    else:
        sigma = config.tsys / math.sqrt(config.bandwidth * config.tau) * config.noise_scale
    return float(sigma), median


def snr_value(median, sigma):
    if sigma == 0:
        return math.inf
    return float(20 * np.log10(median / sigma)) if median > 0 else -math.inf


def add_noise(clean, sigma, seed, scene_index, gamma_index, repeat):
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_index, gamma_index, repeat]))
    noise = (rng.standard_normal(len(clean)) + 1j * rng.standard_normal(len(clean))) / math.sqrt(2)
    return (clean + sigma * noise).astype(np.complex64)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def peak_memory_bytes():
    """Process peak working set, using only the standard library."""
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                 "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                 "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counter), counter.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counter.PeakWorkingSetSize)
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class BandDistribution:
    """Exact medians of all full-aperture samples, never gamma duplicates.

    The full dataset stores ~460 MB of float32 magnitudes and signal/noise
    ratios temporarily. No measurement matrix or per-pixel operator is stored.
    """
    def __init__(self, radii):
        top = max(5.0, math.ceil(float(np.max(radii)) / 5) * 5)
        self.edges = np.arange(0, top + 5, 5, dtype=np.float64)
        self.masks = [(radii >= a) & (radii < b) for a, b in zip(self.edges[:-1], self.edges[1:])]
        self.values = [[] for _ in self.masks]
        self.ratios = [[] for _ in self.masks]
        self.noise_free = False

    def append(self, clean_batch, sigmas):
        magnitude = np.abs(clean_batch)
        self.noise_free |= bool(np.any(np.asarray(sigmas) == 0))
        for i, mask in enumerate(self.masks):
            self.values[i].append(magnitude[:, mask].ravel().copy())
            if not self.noise_free:
                self.ratios[i].append((magnitude[:, mask] / np.asarray(sigmas)[:, None]).astype(np.float32).ravel())

    def report(self):
        bands = []
        for i, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            values = np.concatenate(self.values[i]) if self.values[i] else np.empty(0)
            ratios = np.concatenate(self.ratios[i]) if self.ratios[i] else np.empty(0)
            median = float(np.median(values)) if values.size else None
            ratio = float(np.median(ratios)) if ratios.size and not self.noise_free else None
            bands.append({"radius_min_lambda": float(a), "radius_max_lambda": float(b),
                          "sample_count": int(values.size), "median_abs_v_kelvin": median,
                          "snr_db": float(20 * np.log10(ratio)) if ratio and ratio > 0 else None})
        return {"definition": "Full-aperture noiseless |V|; SNR=20log10(median(|V|/sigma_full)); radial bands, upper edge excluded",
                "noise_free": self.noise_free, "bands": bands}


def prepare(config):
    config.validate()
    inventory = collect_inventory(config.input)
    selected = select_inventory(inventory, config.per_class)
    assignments = split_classes({item["class_name"] for item in inventory}, config.seed)
    positions = y_array(config.n_arm, config.spacing_lambda, config.freq)
    u, v = all_baselines(positions, config.freq)
    plan = aperture_plan(u, v, config.gammas, config.theta, config.n_grid)
    return inventory, selected, assignments, positions, u, v, plan


def generate_dataset(config, dry_run=False, progress=True):
    start = time.perf_counter()
    inventory, selected, assignments, positions, u, v, plan = prepare(config)
    n_meas = len(selected) * len(plan) * config.noise_repeats
    payload = len(selected) * (config.n_grid**2 + sum(p.m for p in plan) * 8 * config.noise_repeats)
    if progress:
        print(f"Scenes={len(selected)} classes={len(assignments)} measurements={n_meas}; array payload ~{payload/1024**3:.3f} GiB", flush=True)
        for aperture in plan:
            print(f"  {aperture.tag}: M={aperture.m} independent={aperture.n_independent} cells={aperture.cells:.6f} null={100*aperture.null_fraction:.6f}%", flush=True)
    if dry_run:
        return {"scenes": len(selected), "measurements": n_meas, "estimated_array_bytes": payload}
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = asdict(config)
    settings["input"], settings["output"] = str(Path(config.input).resolve()), str(output.resolve())
    metadata = {"schema_version": SCHEMA_VERSION, "status": "generating", "config": settings,
                "bt_min": config.bt_min, "bt_max": config.bt_max, "theta": config.theta,
                "n_grid": config.n_grid, "noise_repeats": config.noise_repeats,
                "seed": config.seed, "aperture_atol_lambda": APERTURE_ATOL,
                "aperture_definition": "hypot(u,v) <= gamma*r_max + 1e-9 (float64)",
                "r_max_lambda": float(np.hypot(u, v).max()), "n_antenna": len(positions),
                "class_assignment": assignments, "plan": [p.metadata() for p in plan],
                "source_inventory": inventory, "selected_scenes": [i["scene"] for i in selected],
                "inventory_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
                "null_fraction_definition": "1-min(2*n_independent,n_grid**2)/n_grid**2; lower bound from real-equation budget, not a measured rank",
                "noise_seed": "SeedSequence([seed, full_inventory_scene_index, gamma_index, repeat])",
                "expected_counts": {"gt": len(selected), "measurement_files": len(selected), "measurements": n_meas}}
    save_json(output / "scenes.json", metadata)
    arrays = {"u": u.astype(np.float32), "v": v.astype(np.float32)}
    for aperture in plan:
        suffix = aperture.tag
        arrays[f"indices_{suffix}"] = aperture.indices
        for key in ("m", "n_independent", "n_real_constraints_upper"):
            arrays[f"{key}_{suffix}"] = np.int32(getattr(aperture, key))
        for key in ("gamma", "cells", "null_fraction"):
            arrays[f"{key}_{suffix}"] = np.float32(getattr(aperture, key))
    save_arrays(output / "geometry.npz", arrays)
    for cls in sorted({i["class_name"] for i in selected}):
        (output / "gt" / cls).mkdir(parents=True)
        (output / "meas" / cls).mkdir(parents=True)
    operator = ForwardOperator(config.n_grid, config.theta, u, v)
    distribution = BandDistribution(np.hypot(u, v))
    last_progress = time.perf_counter()
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for offset in range(0, len(selected), config.batch):
            batch = selected[offset:offset + config.batch]
            gray = np.stack([read_u8(Path(config.input) / item["source"], config.n_grid) for item in batch])
            kelvin = decode_gt(gray, config.bt_min, config.bt_max)
            clean_full = operator(kelvin)
            distribution.append(clean_full, [sigma_for(clean, config)[0] for clean in clean_full])
            for bi, item in enumerate(batch):
                scene = item["scene"]
                np.save(output / "gt" / (scene + ".npy"), gray[bi], allow_pickle=False)
                contents = {}
                for gi, aperture in enumerate(plan):
                    clean = clean_full[bi, aperture.indices]
                    sigma, median = sigma_for(clean, config)
                    snr = snr_value(median, sigma)
                    noisy = np.stack([add_noise(clean, sigma, config.seed, item["scene_index"], gi, repeat)
                                      for repeat in range(config.noise_repeats)])
                    contents[f"y_{aperture.tag}"] = noisy
                    contents[f"sigma_{aperture.tag}"] = np.full(config.noise_repeats, sigma, dtype=np.float32)
                    contents[f"snr_db_{aperture.tag}"] = np.full(config.noise_repeats, snr, dtype=np.float32)
                    for repeat in range(config.noise_repeats):
                        writer.writerow({"scene": scene, "file": f"meas/{scene}.npz", "class_name": item["class_name"],
                                         "split": assignments[item["class_name"]], "scene_index": item["scene_index"],
                                         "repeat": repeat, "sigma": sigma, "median_abs_v": median, "snr_db": snr,
                                         **{key: getattr(aperture, key) for key in ("gamma", "tag", "m", "n_independent",
                                            "n_real_constraints_upper", "cells", "null_fraction")}})
                save_arrays(output / "meas" / (scene + ".npz"), contents)
            done = min(offset + len(batch), len(selected))
            now = time.perf_counter()
            if progress and (now - last_progress >= 10 or done == len(selected)):
                elapsed = now - start
                print(f"Generated {done}/{len(selected)} scenes ({done/elapsed:.1f}/s), elapsed {elapsed:.1f}s", flush=True)
                last_progress = now
    report = distribution.report()
    save_json(output / "distribution.json", report)
    if progress:
        for band in report["bands"]:
            print("Band", band, flush=True)
    metadata.update(status="complete", generation_seconds=time.perf_counter() - start,
                    generation_peak_memory_bytes=peak_memory_bytes())
    save_json(output / "scenes.json", metadata)
    return metadata


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    for name, kind, default in [("theta", float, 1.05), ("n-grid", int, 256), ("n-arm", int, 21),
         ("spacing-lambda", float, .875), ("freq", float, 1.4135e9), ("tsys", float, 350.),
         ("bandwidth", float, 19e6), ("tau", float, 1.2), ("noise-scale", float, 1.),
         ("noise-repeats", int, 1), ("per-class", int, 40), ("seed", int, 20261022),
         ("batch", int, 8), ("bt-min", float, 150.), ("bt-max", float, 320.)]:
        result.add_argument("--" + name, type=kind, default=default)
    result.add_argument("--gammas", default="1.0,0.75,0.5,0.35")
    result.add_argument("--snr-db", type=float)
    result.add_argument("--dry-run", action="store_true")
    return result


def main():
    args = vars(parser().parse_args())
    dry_run = args.pop("dry_run")
    args["gammas"] = tuple(float(x.strip()) for x in args["gammas"].split(","))
    try:
        metadata = generate_dataset(Config(**args), dry_run=dry_run)
        print(json.dumps({k: metadata[k] for k in ("status", "generation_seconds", "generation_peak_memory_bytes") if k in metadata}), flush=True)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Generation failed: {error}") from error


if __name__ == "__main__":
    main()
