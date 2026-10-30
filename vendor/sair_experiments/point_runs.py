"""Independent source ensembles, method-unit calibration and frozen evaluation."""

import importlib
from pathlib import Path
import time
import numpy as np
from .design import SharedNetwork, FAMILIES, fingerprint
from .planning import point_plan
from .points import (source_scene, experiment_noise, dirty_components, clean, matched_filter,
                     blind_peaks, assess_peaks)
from .runtime import write_json, physical_sigma, physical_memory, resident_memory
from .runtime import IMPLEMENTATION_SHA256, FORMAL_EXECUTION
from .solvers import Problem, solve, solve_path
from .evaluation import image_metrics, certify_separation


def network_adapter(spec, checkpoint):
    if not spec or not checkpoint:
        raise ValueError("network requires an explicit module:function and a shared checkpoint")
    module, function = spec.split(":", 1)
    # Factory receives the checkpoint path, returns predict(packet).
    return SharedNetwork(getattr(importlib.import_module(module), function)(checkpoint), checkpoint)


def point_packet(op, y, sigma, bounds):
    return {"visibility_ri": np.column_stack((y.real, y.imag)).astype("float32"),
            "mask": np.ones(len(y), bool), "uv": np.column_stack((op.u, op.v)).astype("float32"),
            "gamma": 1., "sigma": float(sigma),
            "physics": {"theta": op.theta, "n_grid": op.n, "bt_min": bounds[0], "bt_max": bounds[1]}}


def point_configs(op, phase):
    plan = point_plan(op)
    if phase == "formal":
        configs = list(plan["configurations"])
        for i, sources in enumerate((0, 1)):
            configs.append(dict(config_id=200+i, sources=sources, center=(0., 0.),
                                noise_repeats=650, family=FAMILIES["controls"], control=True))
        for i, center in enumerate(((-op.theta/8, 0.), (op.theta/8, op.theta/8))):
            configs.append(dict(config_id=300+i, sources=1, center=center, noise_repeats=10,
                                family=FAMILIES["offset_single"], offset_single=True))
    elif phase == "calibration":
        configs = [dict(config_id=i, sources=s, center=(0., 0.), noise_repeats=650,
                        family=FAMILIES["calibration"]) for i, s in enumerate((0, 1))]
    elif phase == "tuning":
        configs = []
        source = plan["configurations"]
        for i in range(120):
            c = dict(source[(i*17) % len(source)]) if i < 60 else dict(sources=1 if i < 90 else 0, center=(0., 0.))
            c.update(config_id=400+i, noise_repeats=1, family=FAMILIES["l1_tuning"], include_noiseless=False)
            configs.append(c)
    else:
        raise ValueError("unknown point phase")
    return configs


def observations(op, sigma, bounds, phase):
    """Compute direct-sum truth visibility ONCE per source configuration."""
    for c in point_configs(op, phase):
        truth, positions, flux = source_scene(op, center=c.get("center", (0., 0.)),
                        degrees=c.get("degrees", 0.), ratio=c.get("ratio", 1.),
                        distance=c.get("distance", 0.), sources=c["sources"], bounds=bounds)
        clean_y = op.reference(truth)
        for repeat in range(c["noise_repeats"]):
            y = experiment_noise(clean_y, sigma, c["family"], c["config_id"], repeat)
            yield c, repeat, y, truth, positions, clean_y, flux
        if c.get("include_noiseless"):
            yield c, -1, clean_y.copy(), truth, positions, clean_y, flux


def residual_stop_calibration(op, sigma, bounds):
    # Blank means no contrast; original weighted background is subtracted.
    clean_y = op.reference(np.full((op.n, op.n), bounds[0]))
    maxima = []
    for repeat in range(650):
        y = experiment_noise(clean_y, sigma, FAMILIES["calibration"], 0, repeat)
        maxima.append(float(dirty_components(op, y-clean_y).max()))
    return float(np.quantile(maxima, .99, method="higher"))


def reconstruct(method, op, y, sigma, bounds, lam, tolerance, max_calls,
                task_id, thresholds, network=None):
    start = time.perf_counter()
    if method == "l1":
        problem = Problem(op, y, sigma, bounds, task_id)
        result = solve(problem, "l1", lam, tolerance=tolerance, max_calls=max_calls)
        prediction = problem.temperature(result.z)
        image = prediction-bounds[0]
        record = {"converged": result.converged, "status": result.status,
                  "calls": result.calls, "iterations": result.iterations,
                  "initial_background_forward_calls": 1,
                  "primal": result.primal, "lower_bound": result.lower, "relative_gap": result.relative_gap,
                  "total_mass_z": float(result.z.sum()), "effective_support_gt_1percent": int(np.sum(result.z > .01)),
                  "weighted_contrast_flux": float(np.sum(op.w*image))}
        unit = "temperature_contrast"
    elif method == "clean":
        stop = thresholds.get("clean_residual_stop")
        result = clean(op, y, bounds[0], stop_threshold=stop)
        image = result.components
        # This diagnostic is the component model divided by quadrature weight,
        # NOT a restored image or a temperature map for the RMSE leaderboard.
        prediction = None
        record = {"converged": result.status == "threshold_reached", "status": result.status,
                  "iterations": result.iterations, "component_flux": float(image.sum()),
                  "calls": {"forward": 1, "adjoint": result.iterations+(result.status == "threshold_reached")}}
        unit = "component_integral"
    elif method == "matched_filter":
        image = matched_filter(op, y, sigma, bounds[0])
        prediction = None
        record = {"converged": True, "status": "template_scores_computed", "template_tail_sigma": 8}
        unit = "standardized_score"
    elif method == "network":
        if network is None:
            raise ValueError("shared network not supplied")
        prediction = network(point_packet(op, y, sigma, bounds))
        image = prediction-bounds[0]
        record = {"converged": True, "status": "inference_complete", "checkpoint_sha256": network.checkpoint_sha256}
        unit = "temperature_contrast"
    else:
        raise ValueError("unknown source method")
    record.update(seconds_reconstruction=time.perf_counter()-start, method=method, detection_units=unit)
    return image, prediction, record


def run_points(reader, op, out, phase, method, lam=1., tolerance=1e-5, max_calls=10000,
               thresholds=None, network=None, limit=None):
    bounds = reader.metadata["bt_min"], reader.metadata["bt_max"]
    sigma = physical_sigma(reader)
    thresholds = dict(thresholds or {})
    if method == "clean" and phase == "calibration":
        thresholds["clean_residual_stop"] = residual_stop_calibration(op, sigma, bounds)
    key = method+"_"+({"clean": "component_integral", "matched_filter": "standardized_score"}.get(method, "temperature_contrast"))
    if phase == "formal" and key not in thresholds:
        raise ValueError("method-specific frozen detection threshold is missing")
    records, counts, controls, calib_blank, calib_second = [], {}, {0: 0, 1: 0}, [], []
    expected = 14028 if phase == "formal" else 1300
    start = time.perf_counter()
    for number, (c, repeat, y, truth, positions, clean_y, flux) in enumerate(observations(op, sigma, bounds, phase)):
        if limit is not None and number >= limit:
            break
        child = Path(out)/f"task_{number:05d}"; child.mkdir()
        task_start = time.perf_counter()
        try:
            image, prediction, record = reconstruct(method, op, y, sigma, bounds, lam, tolerance,
                        max_calls, f'{phase}/{c["family"]}/{c["config_id"]}/{repeat}', thresholds, network)
            threshold = thresholds.get(key, 0.)
            peaks = blind_peaks(image, threshold)
            assessment = assess_peaks(image, peaks, positions, require_valley=method != "matched_filter")
            record.update(config=c, repeat=repeat, gamma=1., phase=phase, scene_type="synthetic", batch=1, workers=1,
                          tolerance=tolerance if method == "l1" else None, **assessment,
                          rss_bytes=resident_memory(), source_reference_flux=flux,
                          sigma_ref=float(sigma), noise_sigma=float(sigma) if repeat >= 0 else 0.,
                          temperature_bounds=list(bounds), operator_sha256=op.identity,
                          implementation_sha256=IMPLEMENTATION_SHA256)
            arrays = {"detection_map": image, "truth": truth, "truth_positions": positions, "detected_positions": peaks}
            if prediction is not None:
                arrays["temperature"] = prediction
                record.update(image_metrics(prediction, truth, bounds[1]-bounds[0]))
            np.savez(child/"maps.npz", **arrays)
            if phase == "calibration":
                raw = blind_peaks(image, 0.)
                heights = [float(image[tuple(p.astype(int))]) for p in raw]
                if c["sources"] == 0:
                    calib_blank.append(heights[0] if heights else 0.)
                else:
                    calib_second.append(heights[1] if len(heights) > 1 else 0.)
            elif c.get("confirmatory") and repeat >= 0:
                k = c["degrees"], c["distance"]
                counts[k] = counts.get(k, 0)+int(assessment["separated"] and record["converged"])
            elif c.get("control"):
                controls[c["sources"]] += int(len(peaks) > (0 if c["sources"] == 0 else 1) or not record["converged"])
            record["seconds_end_to_end"] = time.perf_counter()-task_start
            write_json(child/"result.json", record)
            records.append(record)
        except Exception as exc:
            failure = {"status": "failed", "converged": False, "error": repr(exc), "config": c, "repeat": repeat}
            np.savez(child/"failure_input.npz", truth=truth, visibility=y, clean_visibility=clean_y, truth_positions=positions)
            write_json(child/"failure.json", failure)
            records.append(failure)
            if c.get("confirmatory") and repeat >= 0:
                counts.setdefault((c["degrees"], c["distance"]), 0)
            if c.get("control"):
                controls[c["sources"]] += 1
        if number % 25 == 0:
            write_json(Path(out)/"progress.json", {"completed": len(records), "expected": expected})
            print(f"{phase}/{method}: {len(records)}/{expected}", flush=True)
    complete = len(records) == expected
    summary = {"method": method, "phase": phase, "records": records, "complete": complete,
               "execution": dict(FORMAL_EXECUTION),
               "seconds_end_to_end": time.perf_counter()-start, "expected": expected,
               "failed_or_unconverged": sum(not r["converged"] for r in records),
               "certificates": {}, "thresholds": thresholds}
    if phase == "calibration" and complete and not summary["failed_or_unconverged"]:
        candidates = sorted(set([0., *calib_blank, *calib_second]))
        for threshold in candidates:
            a = np.mean(np.array(calib_blank) > threshold)
            b = np.mean(np.array(calib_second) > threshold)
            if a <= .01 and b <= .01:
                thresholds[key] = threshold
                summary.update(empirical_false_blank=float(a), empirical_false_split=float(b))
                break
        write_json(Path(out)/"thresholds.json", thresholds)
    if phase == "formal" and complete and method != "matched_filter":
        for angle in (0, 15, 30):
            subset = {d: k for (a, d), k in counts.items() if a == angle}
            from .points import dirty_fwhm
            summary["certificates"][str(angle)] = certify_separation(subset, dirty_fwhm(op, angle), controls[0], controls[1])
    summary["control_counts"] = controls
    write_json(Path(out)/"point_report.json", summary)
    return summary


def tune_l1(reader, op, out, tolerance=1e-5, max_calls=10000, thresholds=None):
    """Separate 120-observation L1 validation set, descending paths at each stage.

    Validation controls set a candidate-specific provisional threshold before
    ranking separation. Independent 650+650 calibration remains mandatory.
    """
    bounds = reader.metadata["bt_min"], reader.metadata["bt_max"]
    sigma = physical_sigma(reader)
    data = list(observations(op, sigma, bounds, "tuning"))
    path_records, ranking, screen_ranking = [], [], None
    coarse = np.logspace(-4, 4, 9)
    for phase in ("coarse", "coarse_formal_audit", "fine"):
        if phase == "coarse":
            candidates = coarse.copy()
        elif phase == "coarse_formal_audit":
            candidates = coarse.copy()
        else:
            winner = ranking[0]["lambda"]
            candidates = np.geomspace(winner/3, winner*3, 7)
        stage_rows, extension = [], 0
        while True:
            for i, (c, repeat, y, truth, positions, _, _) in enumerate(data):
                problem = Problem(op, y, sigma, bounds, f'l1_tuning/{c["config_id"]}/{repeat}')
                start = time.perf_counter()
                tol = 1e-3 if phase == "coarse" else tolerance
                lambda_path = solve_path(problem, "l1", candidates, tolerance=tol, max_calls=max_calls)
                child = Path(out)/f"{phase}_extension{extension}_{i:03d}"; child.mkdir()
                rows = []
                for j, (lam, result) in enumerate(lambda_path):
                    prediction = problem.temperature(result.z)
                    contrast = prediction-bounds[0]
                    peaks = blind_peaks(contrast, 0.)
                    heights = [float(contrast[tuple(p.astype(int))]) for p in peaks]
                    map_file = child/f"lambda_{j:02d}.npz"
                    np.savez(map_file, temperature=prediction, truth_positions=positions)
                    row = {"lambda": lam, "rmse": image_metrics(prediction, truth, problem.delta)["rmse"],
                           "converged": result.converged, "source_count": c["sources"],
                           "total_mass_z": float(result.z.sum()), "effective_support_gt_1percent": int(np.sum(result.z > .01)),
                           "relative_gap": result.relative_gap, "calls": result.calls,
                           "largest_peak": heights[0] if heights else 0.,
                           "second_peak": heights[1] if len(heights) > 1 else 0., "map_file": str(map_file)}
                    rows.append(row); stage_rows.append(row)
                write_json(child/"results.json", rows)
                path_records.append({"method": "l1", "gamma": 1., "phase": phase, "scene_type": "synthetic", "batch": 1,
                                     "tolerance": tol, "unit": "whole_lambda_paths",
                                     "lambdas": list(map(float, candidates)),
                                     "converged": all(r["converged"] for r in rows),
                                     "seconds_end_to_end": time.perf_counter()-start})
                write_json(Path(out)/"path_timings.json", path_records)
            ranking = []
            for lam in sorted({r["lambda"] for r in stage_rows}):
                subset = [r for r in stage_rows if r["lambda"] == lam]
                if len(subset) != 120 or not all(r["converged"] for r in subset):
                    continue
                # With only 30 controls per kind, the <=1% empirical target
                # means zero errors. This threshold is validation-only; the
                # independent 650+650 calibration is still required later.
                threshold = max([r["largest_peak"] for r in subset if r["source_count"] == 0]
                               +[r["second_peak"] for r in subset if r["source_count"] == 1])
                successes = 0
                for r in subset:
                    if r["source_count"] != 2:
                        continue
                    with np.load(r["map_file"], allow_pickle=False) as saved:
                        image = saved["temperature"]-bounds[0]
                        successes += int(assess_peaks(image, blind_peaks(image, threshold), saved["truth_positions"])["separated"])
                ranking.append({"lambda": lam, "separation_successes": successes,
                                "rmse": float(np.mean([r["rmse"] for r in subset])),
                                "validation_threshold": threshold})
            ranking.sort(key=lambda r: (-r["separation_successes"], r["rmse"], r["lambda"]))
            if not ranking:
                write_json(Path(out)/"tuning_report.json", {"status": "failed_convergence", "stage": phase})
                return
            smallest, largest = min(r["lambda"] for r in stage_rows), max(r["lambda"] for r in stage_rows)
            winner = ranking[0]["lambda"]
            boundary = winner in (smallest, largest)
            if not boundary:
                break
            if extension == 3:
                write_json(Path(out)/"tuning_report.json", {"status": "search_insufficient", "stage": phase, "ranking": ranking})
                return
            extension += 1
            candidates = np.array([winner/10 if winner == smallest else winner*10])
        if phase == "coarse":
            coarse = np.array(sorted({r["lambda"] for r in stage_rows}))
            screen_ranking = ranking
        write_json(Path(out)/f"{phase}_ranking.json", ranking)
    write_json(Path(out)/"tuning_report.json", {"status": "validation_selected_not_frozen", "ranking": ranking,
               "lambda": ranking[0]["lambda"], "tolerance": tolerance,
               "screening_ranking": screen_ranking, "note": "independent 650+650 calibration and full precision/resource gate required"})
