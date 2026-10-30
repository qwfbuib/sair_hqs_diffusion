"""Result persistence, explicit pilot gates, timing groups, and resource limits."""

import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import hashlib
import numpy as np

from .design import PROTOCOL_VERSION, BASE_TASKS, fingerprint
from .operators import dataset_operator
from .solvers import Problem, solve, solve_path, FINAL_CERTIFICATE_CALLS
from .evaluation import image_metrics, observation_metrics

FORMAL_MAX_CALLS = 10000
# These describe the current compare/points executors, not the fastest pilot.
FORMAL_EXECUTION = {"cpu_workers": 1, "network_batch": 1, "network_api": "single_packet"}


def implementation_digest():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    for name in ("array.py", "operator.py", "reader.py"):
        digest.update((root.parent/"sair_dataset"/name).read_bytes())
    return digest.hexdigest()


IMPLEMENTATION_SHA256 = implementation_digest()


def physical_sigma(reader):
    c = reader.metadata["config"]
    # A noiseless diagnostic keeps the positive physical normalization.
    return c["tsys"]/np.sqrt(c["bandwidth"]*c["tau"])*(c["noise_scale"] if c["noise_scale"] > 0 else 1.)


def environment():
    versions = {}
    for name in ("numpy", "scipy", "Pillow", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
            "logical_cpus": os.cpu_count(), "physical_memory_bytes": physical_memory()}


def physical_memory():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)]+[(name, ctypes.c_ulonglong) for name in
                        ("physical", "available", "page", "available_page", "virtual", "available_virtual", "extended")]
        value = Memory(); value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("cannot read physical memory")
        return value.physical
    return os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")


def resident_memory():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)]+[(name, ctypes.c_size_t) for name in
                        ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        value = Counters(); value.cb = ctypes.sizeof(value)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        get_info.restype = ctypes.c_int
        if not get_info(get_process(), ctypes.byref(value), value.cb):
            raise OSError("cannot read process RSS")
        return value.peak_working
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    # NaN/Infinity are made explicit strings instead of invalid JSON numbers.
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        return v
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def persist_result(out, result, problem, metadata, truth=None, clean=None):
    arrays = {"temperature": problem.temperature(result.z), "z": result.z}
    for tol, snapshot in result.checkpoints.items():
        arrays["z_tol_"+tol] = snapshot["z"]
    np.savez(Path(out)/"prediction.npz", **arrays)
    record = {**metadata, "protocol_version": PROTOCOL_VERSION, "implementation_sha256": IMPLEMENTATION_SHA256, "status": result.status,
              "converged": result.converged, "relative_gap": result.relative_gap,
              "primal": result.primal, "lower_bound": result.lower, "iterations": result.iterations,
              "calls": result.calls, "seconds_solver": result.seconds, "trace": result.trace,
              "initial_background_forward_calls": 1, "rss_bytes": resident_memory(),
              "sigma_ref": problem.sigma_ref, "temperature_bounds": [problem.lo, problem.hi],
              "operator_sha256": problem.op.identity,
              "total_mass_unweighted_z": float(result.z.sum()),
              "weighted_contrast_flux": float(np.sum(problem.op.w*problem.delta*result.z)),
              "effective_support_gt_1percent": int(np.count_nonzero(result.z > .01))}
    record.update(observation_metrics(problem.op, arrays["temperature"], problem.y, problem.sigma_ref, clean))
    if truth is not None:
        record.update(image_metrics(arrays["temperature"], truth, problem.delta))
    write_json(Path(out)/"result.json", record)
    return record


def natural_task(reader, scene, gamma, method, lam, out, tolerance=1e-5, max_calls=10000, repeat=0):
    start = time.perf_counter()
    obs = reader.load(scene, gamma, repeat)
    op = dataset_operator(reader, gamma)
    problem = Problem(op, obs["y"], physical_sigma(reader),
                      (reader.metadata["bt_min"], reader.metadata["bt_max"]), f"{scene}/{gamma}/{repeat}")
    setup_seconds = time.perf_counter()-start
    result = solve(problem, method, lam, tolerance=tolerance, max_calls=max_calls)
    truth = reader.truth(scene)
    record = persist_result(out, result, problem,
                            {"scene": scene, "gamma": gamma, "repeat": repeat, "method": method,
                             "lambda": lam, "tolerance": tolerance, "phase": "validation_pilot",
                             "scene_type": "natural", "batch": 1, "setup_seconds": setup_seconds,
                             "observation_key": problem.key}, truth, op(truth))
    record["seconds_end_to_end"] = time.perf_counter()-start
    write_json(Path(out)/"result.json", record)
    return record


def precision_comparison(result, truth, delta, decisions=None):
    """No certificate for a missing checkpoint or uncalibrated detection test."""
    candidate = result.checkpoints.get("0.0001")
    reference = result.checkpoints.get("1e-05")
    if candidate is None or reference is None:
        return {"passed": False, "reason": "candidate or converged strict checkpoint missing"}
    difference = delta*float(np.sqrt(np.mean((candidate["z"]-reference["z"])**2)))
    # Caller passes normalized truth; this comparison is independent of Tmin.
    a = delta*float(np.sqrt(np.mean((candidate["z"]-truth)**2)))
    b = delta*float(np.sqrt(np.mean((reference["z"]-truth)**2)))
    relative = abs(a-b)/max(b, 1e-12)
    numerical_pass = difference <= .01 and relative <= .001
    return {"prediction_difference_rmse_k": difference, "main_rmse_relative_difference": relative,
            "numerical_pass": numerical_pass, "detection_consistent": decisions,
            "passed": bool(numerical_pass and decisions is True),
            "reason": "detection audit pending" if decisions is None else "checked"}


def precision_call_limits(result):
    """Checkpoints must be reachable with the formal solver's reserved calls."""
    available = FORMAL_MAX_CALLS - FINAL_CERTIFICATE_CALLS
    return {str(tol): bool((checkpoint := result.checkpoints.get(str(tol))) is not None
                          and 0 <= checkpoint.get("calls", float("inf")) <= available
                          and checkpoint.get("relative_gap", float("inf")) <= tol)
            for tol in (1e-4, 1e-5)}


def wall_budget(records, workload, fixed_overheads=None):
    """Only completed, converged matching jobs justify a formal throughput.

    Workload rows specify method/gamma/phase/scene_type/tolerance/batch plus
    task count. Tuning records must time an entire path, with count in paths.
    Missing groups remain unmeasured rather than inheriting another cost.
    """
    keys = ("method", "gamma", "phase", "scene_type", "tolerance", "batch")
    groups = []
    for row in workload:
        if row.get("batch", 1) != 1 or row.get("workers", 1) != 1:
            raise ValueError("current formal executor requires workers=1 and batch=1")
        matched = [r for r in records if all(r.get(k) == row.get(k) for k in keys)
                   and r.get("workers", 1) == 1 and r.get("within_memory_budget", True)
                   and (row["method"] != "network" or r.get("execution_api") == "single_packet")
                   and r.get("unit", "tasks") == row.get("unit", "tasks")
                   and ("lambdas_per_path" not in row or len(r.get("lambdas", [])) == row["lambdas_per_path"])]
        completed = [r for r in matched if r.get("converged") and r.get("seconds_end_to_end", 0) > 0]
        entry = {**row, "measured_cohorts": len(matched),
                 "measured_tasks": sum(r.get("tasks", 1) for r in matched),
                 "completed_tasks": sum(r.get("tasks", 1) for r in completed)}
        if not completed:
            entry.update(status="unmeasured", estimated_seconds=None)
        else:
            throughputs = [r for r in completed if r.get("completed_throughput", 0) > 0]
            if throughputs:
                selected = max(throughputs, key=lambda r: r["completed_throughput"])
                seconds = 1/selected["completed_throughput"]
                entry.update(status="measured", measurement_unit="completed_serial_throughput",
                             workers=selected["workers"], mean_seconds=seconds,
                             estimated_seconds=float(row["count"]*seconds),
                             p50_seconds=None, p95_seconds=None, p95_cost_scenario_seconds=None)
                groups.append(entry)
                continue
            timings = np.array([r["seconds_end_to_end"] for r in completed])
            entry.update(status="measured", mean_seconds=float(timings.mean()),
                         p50_seconds=float(np.median(timings)), p95_seconds=float(np.quantile(timings, .95)),
                         estimated_seconds=float(timings.mean()*row["count"]),
                         p95_cost_scenario_seconds=float(np.quantile(timings, .95)*row["count"]))
        groups.append(entry)
    complete = all(g["status"] == "measured" for g in groups)
    overheads = fixed_overheads or {}
    return {"groups": groups, "complete": complete, "execution": dict(FORMAL_EXECUTION), "fixed_overheads_seconds": overheads,
            "total_seconds": sum(g["estimated_seconds"] for g in groups)+sum(overheads.values()) if complete else None,
            "note": "serial workers=1/batch=1 execution only; parallel/batch pilots are diagnostic; evaluation and precompute costs must be added separately; p95 scenarios are not total-time confidence bounds"}


def formal_gate(accuracy, resource, frozen):
    tolerances = frozen.get("tolerances", {})
    def approved(which):
        tol = tolerances.get(which)
        within_limit = (accuracy.get("formal_call_limit") == FORMAL_MAX_CALLS
                        and accuracy.get("within_formal_call_limit", {}).get(which, {}).get(str(tol)) is True)
        return within_limit and (tol == 1e-5 or (tol == 1e-4 and accuracy.get(which+"_passed") is True))
    required_thresholds = {"l1_temperature_contrast", "clean_component_integral", "clean_residual_stop",
                           "network_temperature_contrast", "matched_filter_standardized_score"}
    thresholds = frozen.get("thresholds", {})
    valid_thresholds = required_thresholds <= set(thresholds) and all(
        np.isfinite(thresholds[k]) and thresholds[k] >= 0 for k in required_thresholds if k in thresholds)
    required = {"natural_precision": approved("natural"),
                "l1_precision": approved("l1"),
                "strict_references": accuracy.get("all_references_converged") is True,
                "hard_task_stability": accuracy.get("five_hard_tasks_stable") is True,
                "complete_precision_tasks": accuracy.get("natural_count") == 20 and accuracy.get("synthetic_count") == 30,
                "resource_pilot": resource.get("passed") is True,
                "wall_budget_measured": resource.get("wall_budget_complete") is True,
                "execution_matches_budget": resource.get("execution") == frozen.get("execution") == FORMAL_EXECUTION,
                "formal_call_limit": frozen.get("max_calls") == FORMAL_MAX_CALLS,
                "memory_limit": resource.get("peak_total_rss_bytes", float("inf")) <= .7*physical_memory(),
                "frozen_configuration": frozen.get("status") == "frozen",
                "audited_manifest": bool(accuracy.get("manifest_sha256")) and accuracy.get("manifest_sha256") == frozen.get("manifest_sha256"),
                "audited_lambdas": bool(accuracy.get("lambdas")) and accuracy.get("lambdas") == frozen.get("lambdas"),
                "shared_network_checkpoint": bool(frozen.get("network_checkpoint_sha256")),
                "calibrated_thresholds": valid_thresholds}
    return {"allowed": all(required.values()), "checks": required,
            "blocking_reasons": [k for k, v in required.items() if not v]}
