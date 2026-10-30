"""Structural checks, independent physics references, and noise replay."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .array import all_baselines, aperture_plan, y_array
from .generate import Config, add_noise, peak_memory_bytes, read_u8, save_json, sigma_for, snr_value, split_classes
from .operator import ForwardOperator, decode_gt, pixel_weights
from .reader import DatasetReader


def require(condition, message):
    if not condition:
        raise ValueError(message)


def direct_reference(x, weights, u, v, theta, indices):
    """Independent selected-row sum, never a full operator matrix.

    Only eight diagnostic rows are checked, one at a time; production
    forward sampling is vectorized over all measured rows.
    """
    n = len(x)
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * theta / n
    values = []
    for index in indices:
        phase = np.exp(-2j * np.pi * (u[index] * axis[:, None] + v[index] * axis[None, :]))
        values.append(np.sum(x * weights * phase))
    return np.array(values)


def audit_dataset(root, compare_pilot=None, samples_per_class=3, progress=True, workers=4):
    started = time.perf_counter()
    root = Path(root)
    reader = DatasetReader(root)
    meta = reader.metadata
    settings = dict(meta["config"])
    settings["input"], settings["output"] = Path(settings["input"]), Path(settings["output"])
    settings["gammas"] = tuple(settings["gammas"])
    config = Config(**settings)
    positions = y_array(config.n_arm, config.spacing_lambda, config.freq)
    u, v = all_baselines(positions, config.freq)
    plan = aperture_plan(u, v, config.gammas, config.theta, config.n_grid)
    require(meta["plan"] == [p.metadata() for p in plan], "aperture metadata differs from float64 geometry")
    require(np.array_equal(reader.geometry["u"], u.astype(np.float32)) and
            np.array_equal(reader.geometry["v"], v.astype(np.float32)), "stored coordinates differ")
    require(reader.geometry["u"].dtype == np.float32 and reader.geometry["v"].dtype == np.float32,
            "coordinates must be float32 on disk")
    for aperture in plan:
        tag = aperture.tag
        require(reader.geometry[f"indices_{tag}"].dtype == np.int32 and
                np.array_equal(reader.geometry[f"indices_{tag}"], aperture.indices), f"bad aperture indices: {tag}")
        for field in ("m", "n_independent", "n_real_constraints_upper", "gamma", "cells", "null_fraction"):
            stored = reader.geometry[f"{field}_{tag}"]
            dtype = np.int32 if field in ("m", "n_independent", "n_real_constraints_upper") else np.float32
            require(stored.shape == () and stored.dtype == dtype and
                    stored.item() == dtype(getattr(aperture, field)).item(), f"bad geometry scalar: {field}_{tag}")
    if config.n_arm == 21 and config.spacing_lambda == .875 and config.gammas == (1., .75, .5, .35):
        require([p.m for p in plan] == [1830, 1635, 1098, 753], "default aperture counts differ")
        require([p.n_independent for p in plan] == [1260, 1065, 537, 267], "default independent counts differ")
    inventory = meta["source_inventory"]
    index = {item["scene"]: item for item in inventory}
    scenes = meta["selected_scenes"]
    require(len(index) == len(inventory) and len(set(scenes)) == len(scenes), "duplicate scene identities")
    require(meta["class_assignment"] == split_classes({i["class_name"] for i in inventory}, config.seed),
            "class split differs from seeded largest-remainder split")
    expected_scene_set = set(scenes)
    actual_gt = {p.relative_to(root / "gt").with_suffix("").as_posix() for p in (root / "gt").rglob("*.npy")}
    actual_meas = {p.relative_to(root / "meas").with_suffix("").as_posix() for p in (root / "meas").rglob("*.npz")}
    require(actual_gt == expected_scene_set and actual_meas == expected_scene_set, "file identities/counts differ")
    rows_by_scene = defaultdict(dict)
    with (root / "summary.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["tag"], int(row["repeat"]))
            require(key not in rows_by_scene[row["scene"]], "duplicate summary measurement")
            rows_by_scene[row["scene"]][key] = row
    require(set(rows_by_scene) == expected_scene_set, "summary scene identities differ")
    require(meta["expected_counts"] == {"gt": len(scenes), "measurement_files": len(scenes),
                                        "measurements": len(scenes)*len(plan)*config.noise_repeats}, "expected counts differ")
    by_class = defaultdict(list)
    for scene in scenes:
        by_class[index[scene]["class_name"]].append(scene)
    sample_scenes = set()
    for group in by_class.values():
        positions_sample = np.linspace(0, len(group)-1, min(samples_per_class, len(group)), dtype=int)
        sample_scenes.update(group[i] for i in positions_sample)
    operator = ForwardOperator(config.n_grid, config.theta, u, v)
    constant = ForwardOperator(config.n_grid, config.theta, np.array([0.]), np.array([0.]))(
        np.full((config.n_grid, config.n_grid), 300.))
    require(abs(constant[0] - 300) / 300 < 1e-6, "zero-baseline normalization failed")
    direct_max_error = 0.
    replay_measurements = 0
    expected_fields = {f"{field}_{p.tag}" for p in plan for field in ("y", "sigma", "snr_db")}
    require(workers >= 1 and samples_per_class >= 1, "workers and samples_per_class must be positive")

    def check_scene(scene):
        scene_max_error = 0.
        scene_replays = 0
        item = index[scene]
        truth = np.load(root / "gt" / (scene + ".npy"), allow_pickle=False)
        require(truth.dtype == np.uint8 and truth.shape == (config.n_grid, config.n_grid), f"bad truth: {scene}")
        measurement_path = root / "meas" / (scene + ".npz")
        with zipfile.ZipFile(measurement_path) as archive:
            require(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()), "compressed measurement found")
        with np.load(measurement_path, allow_pickle=False) as archive:
            require(set(archive.files) == expected_fields, f"bad fields: {scene}")
            require(len(rows_by_scene[scene]) == len(plan)*config.noise_repeats, "summary row count differs")
            clean_full = None
            if scene in sample_scenes:
                source_truth = read_u8(config.input / item["source"], config.n_grid)
                require(np.array_equal(truth, source_truth), f"source-to-GT mismatch: {scene}")
                x = decode_gt(truth, config.bt_min, config.bt_max)
                clean_full = operator(x)
                chosen = np.unique(np.r_[np.linspace(0, len(u)-1, 6, dtype=int), np.argmin(np.hypot(u,v)), np.argmax(np.hypot(u,v))])
                direct = direct_reference(x, pixel_weights(config.n_grid, config.theta), u, v, config.theta, chosen)
                err = float(np.max(np.abs(clean_full[chosen] - direct)))
                scene_max_error = err
                require(np.allclose(clean_full[chosen], direct, atol=2e-5, rtol=2e-6), f"independent forward reference failed: {scene}")
            for gi, aperture in enumerate(plan):
                tag = aperture.tag
                y, sigmas, snrs = (archive[f"{field}_{tag}"] for field in ("y", "sigma", "snr_db"))
                require(y.dtype == np.complex64 and y.shape == (config.noise_repeats, aperture.m) and
                        np.isfinite(y).all(), f"bad measurement: {scene}/{tag}")
                require(sigmas.dtype == np.float32 and sigmas.shape == (config.noise_repeats,) and
                        np.isfinite(sigmas).all() and np.all(sigmas >= 0), f"bad sigma: {scene}/{tag}")
                require(snrs.dtype == np.float32 and snrs.shape == (config.noise_repeats,) and
                        not np.isnan(snrs).any(), f"bad SNR: {scene}/{tag}")
                for repeat in range(config.noise_repeats):
                    row = rows_by_scene[scene][(tag, repeat)]
                    require(row["file"] == f"meas/{scene}.npz" and row["class_name"] == item["class_name"] and
                            row["split"] == meta["class_assignment"][item["class_name"]] and
                            int(row["scene_index"]) == item["scene_index"], "summary identity/split differs")
                    for key in ("m", "n_independent", "n_real_constraints_upper", "gamma", "cells", "null_fraction"):
                        require(float(row[key]) == getattr(aperture, key), f"summary {key} differs")
                    require(np.float32(row["sigma"]) == sigmas[repeat] and np.float32(row["snr_db"]) == snrs[repeat],
                            "summary and file noise values differ")
                    if clean_full is not None:
                        clean = clean_full[aperture.indices]
                        sigma, median = sigma_for(clean, config)
                        expected = add_noise(clean, sigma, config.seed, item["scene_index"], gi, repeat)
                        require(np.array_equal(y[repeat], expected), f"bitwise noise replay failed: {scene}/{tag}")
                        require(float(row["sigma"]) == sigma and float(row["median_abs_v"]) == median and
                                float(row["snr_db"]) == snr_value(median, sigma), "noise statistics differ from clean data")
                        scene_replays += 1
        return scene_max_error, scene_replays

    # Independent files use independent archive handles; shared geometry and
    # operator phase arrays are read-only. Report aggregation stays ordered.
    last_progress = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for done, (error, replays) in enumerate(executor.map(check_scene, scenes), 1):
            direct_max_error = max(direct_max_error, error)
            replay_measurements += replays
            if progress and (time.perf_counter()-last_progress >= 10 or done == len(scenes)):
                print(f"Audited {done}/{len(scenes)} scenes", flush=True)
                last_progress = time.perf_counter()
    distribution = json.loads((root / "distribution.json").read_text(encoding="utf-8"))
    radii = np.hypot(u,v)
    warnings = []
    for band in distribution["bands"]:
        count = int(((radii >= band["radius_min_lambda"]) & (radii < band["radius_max_lambda"])).sum()) * len(scenes)
        require(band["sample_count"] == count, "band distribution sample count differs")
        if count:
            median = band["median_abs_v_kelvin"]
            require(median is not None and 0 <= median <= config.bt_max * (1+1e-6), "visibility magnitude violates normalized positive-weight bound")
            if band["radius_min_lambda"] == 0 and median < 0.1:
                warnings.append("Low-radius median visibility < 0.1 K: inspect scaling before release")
    matched_pilot_scenes = 0
    if compare_pilot is not None:
        pilot_root = Path(compare_pilot)
        pilot = DatasetReader(pilot_root)
        require(meta["inventory_sha256"] == pilot.metadata["inventory_sha256"] and
                meta["class_assignment"] == pilot.metadata["class_assignment"], "pilot inventory or split differs")
        for key in ("u", "v"):
            require(np.array_equal(reader.geometry[key], pilot.geometry[key]), "pilot geometry differs")
        for scene in pilot.metadata["selected_scenes"]:
            require(scene in expected_scene_set, f"pilot scene absent from full dataset: {scene}")
            require(np.array_equal(np.load(root/"gt"/(scene+".npy")), np.load(pilot_root/"gt"/(scene+".npy"))), "pilot GT differs")
            with np.load(root/"meas"/(scene+".npz")) as full, np.load(pilot_root/"meas"/(scene+".npz")) as small:
                require(set(full.files) == set(small.files) and all(np.array_equal(full[key], small[key]) for key in full.files),
                        f"pilot measurement differs: {scene}")
            matched_pilot_scenes += 1
    sizes = [path.stat().st_size for path in root.rglob("*") if path.is_file()]
    report = {"status": "pass" if not warnings else "review_required", "warnings": warnings,
              "gt_files": len(scenes), "measurement_files": len(scenes),
              "measurements": sum(len(rows) for rows in rows_by_scene.values()),
              "class_split_counts": dict(Counter(meta["class_assignment"].values())),
              "scene_split_counts": dict(Counter(meta["class_assignment"][index[s]["class_name"]] for s in scenes)),
              "physics_reference_scenes": len(sample_scenes), "replayed_measurements": replay_measurements,
              "independent_forward_max_absolute_error_kelvin": direct_max_error,
              "pilot_scenes_bitwise_matched": matched_pilot_scenes,
              "dataset_bytes_before_audit_report": sum(sizes),
              "audit_workers": workers,
              "audit_seconds": time.perf_counter() - started, "audit_peak_memory_bytes": peak_memory_bytes()}
    save_json(root / "audit_report.json", report)
    if progress:
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--compare-pilot", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.workers < 1:
        parser.error("samples-per-class and workers must be >= 1")
    try:
        report = audit_dataset(args.data, args.compare_pilot, args.samples_per_class, workers=args.workers)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Audit failed: {error}") from error
    if report["status"] != "pass":
        raise SystemExit("Audit requires review; do not start full generation")


if __name__ == "__main__":
    main()
