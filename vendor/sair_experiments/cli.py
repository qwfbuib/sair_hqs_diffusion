import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback
import subprocess
import sys
import numpy as np

from sair_dataset.reader import DatasetReader
from .design import make_manifest, fingerprint, BASE_TASKS, PROTOCOL_VERSION
from .evaluation import certification_boundary, image_metrics
from .operators import dataset_operator
from .planning import point_plan, base_workload
from .points import source_scene, experiment_noise, blind_peaks, assess_peaks
from .runtime import (new_output, write_json, environment, natural_task, physical_sigma,
                      precision_comparison, persist_result, resident_memory, physical_memory,
                      wall_budget, formal_gate, precision_call_limits, FORMAL_MAX_CALLS, FORMAL_EXECUTION)
from .solvers import Problem, solve, solve_path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_manifest(reader, path):
    loaded = load_json(path)
    expected = make_manifest(reader)
    if fingerprint(loaded) != fingerprint(expected):
        raise ValueError("manifest differs from the frozen dataset/selection")
    return loaded


def plan_command(args):
    reader = DatasetReader(args.data)
    out = new_output(args.out)
    manifest = make_manifest(reader)
    write_json(out/"manifest.json", manifest)
    write_json(out/"point_plan.json", point_plan(dataset_operator(reader, 1.)))
    write_json(out/"workload.json", base_workload())
    write_json(out/"environment.json", environment())
    write_json(out/"certification_boundary.json", certification_boundary())
    print(json.dumps({"output": str(out.resolve()), "base_tasks": sum(BASE_TASKS.values()),
                      "main": len(manifest["main"]), "robustness": len(manifest["robustness"]),
                      "network_extra": len(manifest["network_extra"])}, ensure_ascii=False))


def solve_command(args):
    out = new_output(args.out)
    write_json(out/"environment.json", environment())
    result = natural_task(DatasetReader(args.data), args.scene, args.gamma, args.method,
                          args.lam, out, args.tolerance, args.max_calls, args.repeat)
    print(json.dumps({k: result[k] for k in ("status", "relative_gap", "seconds_end_to_end", "rmse", "calls")}))


def _task_report(result, problem, truth, out, metadata, threshold=None, positions=None):
    record = persist_result(out, result, problem, metadata, truth, problem.op(truth))
    decisions = True if positions is None else None
    if positions is not None and threshold is not None:
        decisions_list = []
        for key in ("0.0001", "1e-05"):
            if key not in result.checkpoints:
                break
            image = problem.temperature(result.checkpoints[key]["z"])-problem.lo
            peaks = blind_peaks(image, threshold)
            assessment = assess_peaks(image, peaks, positions)
            decisions_list.append((len(peaks), assessment["matched"], assessment["separated"]))
        decisions = len(decisions_list) == 2 and decisions_list[0] == decisions_list[1]
    comparison = precision_comparison(result, (truth-problem.lo)/problem.delta, problem.delta, decisions)
    comparison["within_formal_call_limit"] = precision_call_limits(result)
    comparison["candidate_within_formal_call_limit"] = comparison["within_formal_call_limit"]["0.0001"]
    comparison["reference_within_formal_call_limit"] = comparison["within_formal_call_limit"]["1e-05"]
    comparison["passed"] &= comparison["candidate_within_formal_call_limit"]
    record["precision_comparison"] = comparison
    write_json(out/"result.json", record)
    return record


def precision_command(args):
    reader = DatasetReader(args.data)
    manifest = verified_manifest(reader, args.manifest)
    if not 0 <= args.natural_count <= 20 or not 0 <= args.synthetic_count <= 30:
        raise ValueError("precision design is fixed at 20 natural and 30 synthetic tasks")
    out = new_output(args.out)
    write_json(out/"environment.json", environment())
    records, failures, work, solved = [], [], [], []
    thresholds = load_json(args.thresholds) if args.thresholds else {}
    from sair_dataset.array import gamma_tag
    selection = load_json(args.selection) if args.selection else {"lambdas": {
        m: {gamma_tag(g): args.lam for g in (1., .75, .5, .35)} for m in ("tikhonov", "tv", "l1")}}
    opcache = {}
    sigma = physical_sigma(reader)
    bounds = (reader.metadata["bt_min"], reader.metadata["bt_max"])
    for i, task in enumerate(manifest["precision_natural"][:args.natural_count]):
        for method in ("tikhonov", "tv"):
            work.append(("natural", i, method, task))
    for i in range(args.synthetic_count):
        work.append(("synthetic", i, "l1", None))
    strict_failed = False
    for kind, index, method, task in work:
        target = out/f"{kind}_{index:03d}_{method}"
        target.mkdir()
        try:
            gamma = task["gamma"] if task else 1.
            if gamma not in opcache:
                opcache[gamma] = dataset_operator(reader, gamma)
            op = opcache[gamma]
            positions = None
            if kind == "natural":
                obs = reader.load(task["scene"], gamma, task["repeat"])
                truth = reader.truth(task["scene"])
                y = obs["y"]
            else:
                count = index % 3
                theta = (0, 15, 30)[(index//3) % 3]
                from .points import dirty_fwhm
                distance = (.6, 1., 1.25)[(index//9) % 3]*dirty_fwhm(op, theta)
                centers = ((0., 0.), (-op.theta/8, 0.), (op.theta/8, op.theta/8))
                truth, positions, _ = source_scene(op, degrees=theta, sources=count,
                                                   center=centers[(index//3) % 3], ratio=1.+(index//6) % 2,
                                                   distance=distance, bounds=bounds)
                y = experiment_noise(op.reference(truth), sigma, 2, index, 0)
            p = Problem(op, y, sigma, bounds, f"precision/{kind}/{index}/{gamma}")
            lam = selection["lambdas"][method][gamma_tag(gamma)]
            result = solve(p, method, lam, tolerance=1e-5, max_calls=args.max_calls)
            rec = _task_report(result, p, truth, target,
                               {"phase": "precision_audit", "scene_type": kind, "method": method,
                                "task_index": index, "gamma": gamma, "tolerance": 1e-5,
                                "lambda": lam, "batch": 1, "scene": task["scene"] if task else None},
                               thresholds.get("l1_temperature_contrast"), positions)
            records.append(rec)
            if result.converged:
                solved.append((result.seconds, result, p, truth, positions, target, method, lam))
            # Save progress after EVERY task, including failures and limits.
            write_json(out/"progress.json", {"records": records, "failures": failures})
            print(json.dumps({"task": target.name, "status": result.status,
                              "relative_gap": result.relative_gap, "calls": result.calls,
                              "seconds": result.seconds}, ensure_ascii=False), flush=True)
            if not result.converged:
                strict_failed = True
                if args.stop_on_reference_failure:
                    break
        except Exception as exc:
            failures.append({"task": target.name, "error": repr(exc), "traceback": traceback.format_exc()})
            write_json(target/"failure.json", failures[-1])
            if args.stop_on_reference_failure:
                break
    hard_checks = []
    if len(records) == 70 and not failures and all(r["converged"] for r in records):
        for _, strict, problem, truth, positions, target, method, lam in sorted(solved, key=lambda v: v[0], reverse=True)[:5]:
            tighter = solve(problem, method, lam, tolerance=1e-6, max_calls=40000, state=strict.state)
            difference = problem.delta*float(np.sqrt(np.mean((tighter.z-strict.z)**2)))
            a = image_metrics(problem.temperature(strict.z), truth, problem.delta)["rmse"]
            b = image_metrics(problem.temperature(tighter.z), truth, problem.delta)["rmse"]
            detection_stable = True if positions is None else None
            threshold = thresholds.get("l1_temperature_contrast")
            if positions is not None and threshold is not None:
                def decision(z):
                    image = problem.temperature(z)-problem.lo
                    peaks = blind_peaks(image, threshold)
                    assessment = assess_peaks(image, peaks, positions)
                    return len(peaks), assessment["matched"], assessment["separated"]
                detection_stable = decision(strict.z) == decision(tighter.z)
            checked = {"task": target.name, "converged": tighter.converged,
                       "difference_rmse_k": difference, "main_rmse_relative_difference": abs(a-b)/max(b, 1e-12),
                       "detection_stable": detection_stable, "calls": tighter.calls, "seconds": tighter.seconds}
            checked["passed"] = bool(tighter.converged and difference <= .01 and checked["main_rmse_relative_difference"] <= .001 and detection_stable is True)
            hard_checks.append(checked)
            child = target/"tighter_1e-6"; child.mkdir()
            persist_result(child, tighter, problem, checked, truth)
    natural = [r for r in records if r["scene_type"] == "natural"]
    synthetic = [r for r in records if r["scene_type"] == "synthetic"]
    report = {"protocol_version": PROTOCOL_VERSION, "manifest_sha256": manifest["manifest_sha256"],
              "natural_count": len(natural)//2, "synthetic_count": len(synthetic),
              "natural_passed": len(natural) == 40 and all(r["precision_comparison"]["passed"] for r in natural),
              "l1_passed": len(synthetic) == 30 and all(r["precision_comparison"]["passed"] for r in synthetic),
              "all_references_converged": len(records) == 70 and all(r["converged"] for r in records),
              "five_hard_tasks_stable": len(hard_checks) == 5 and all(r["passed"] for r in hard_checks),
              "hard_task_checks": hard_checks,
              "candidate_tolerance": 1e-4, "fallback_tolerance": 1e-5,
              "formal_call_limit": FORMAL_MAX_CALLS,
              "within_formal_call_limit": {
                  family: {str(tol): len(group) == count and all(
                      r["precision_comparison"]["within_formal_call_limit"][str(tol)] for r in group)
                           for tol in (1e-4, 1e-5)}
                  for family, group, count in (("natural", natural, 40), ("l1", synthetic, 30))},
              "lambdas": selection["lambdas"],
              "strict_reference_failed": strict_failed, "records": records, "failures": failures,
              "formal_ready": False}
    write_json(out/"precision_report.json", report)
    print(json.dumps({k: report[k] for k in ("natural_count", "synthetic_count", "natural_passed", "l1_passed", "all_references_converged")}))


def tune_command(args):
    reader = DatasetReader(args.data)
    manifest = verified_manifest(reader, args.manifest)
    out = new_output(args.out)
    write_json(out/"environment.json", environment())
    op = dataset_operator(reader, args.gamma)
    bounds = (reader.metadata["bt_min"], reader.metadata["bt_max"])
    summary, candidate_ranking, boundary = [], [], False
    coarse = np.logspace(-4, 4, 9)
    coarse_candidates = coarse.copy()
    for stage, scenes, tol in (("coarse", manifest["tuning_coarse"], 1e-3),
                               ("coarse_formal_audit", manifest["tuning_coarse"], args.tolerance),
                               ("fine", manifest["tuning_fine"], args.tolerance),
                               ("final", manifest["tuning_final"], args.tolerance)):
        if stage == "coarse":
            candidates = coarse
        elif stage == "coarse_formal_audit":
            candidates = coarse_candidates
        elif stage == "fine":
            winner = candidate_ranking[0][0]
            candidates = np.geomspace(winner/3, winner*3, 7)
        else:
            candidates = [v[0] for v in candidate_ranking[:max(2, args.final_candidates)]]
        extension = 0
        stage_rows = []
        while True:
            for index, scene in enumerate(scenes):
                target = out/f"{stage}_extension{extension}_{index:03d}"
                target.mkdir()
                start = time.perf_counter()
                obs = reader.load(scene, args.gamma)
                truth = reader.truth(scene)
                p = Problem(op, obs["y"], physical_sigma(reader), bounds, f"tune/{scene}/{args.gamma}")
                path = solve_path(p, args.method, candidates, tolerance=tol, max_calls=args.max_calls)
                rows = []
                for j, (lam, result) in enumerate(path):
                    child = target/f"lambda_{j:02d}"; child.mkdir()
                    row = persist_result(child, result, p,
                                         {"phase": stage, "scene": scene, "gamma": args.gamma,
                                          "method": args.method, "lambda": lam, "tolerance": tol}, truth)
                    rows.append(row); stage_rows.append(row)
                summary.append({"method": args.method, "gamma": args.gamma, "phase": stage,
                                "scene_type": "natural", "batch": 1, "tolerance": tol,
                                "converged": all(r["converged"] for r in rows),
                                "unit": "whole_lambda_paths", "lambdas": list(map(float, candidates)),
                                "seconds_end_to_end": time.perf_counter()-start})
                write_json(out/"path_timings.json", summary)
            ranked = []
            for lam in sorted({r["lambda"] for r in stage_rows}):
                subset = [r for r in stage_rows if r["lambda"] == lam]
                if len(subset) == len(scenes) and all(r["converged"] for r in subset):
                    ranked.append((lam, float(np.mean([r["rmse"] for r in subset]))))
            candidate_ranking = sorted(ranked, key=lambda r: (r[1], r[0]))
            if not candidate_ranking:
                write_json(out/"tuning_report.json", {"status": "failed_convergence", "stage": stage})
                return
            winner = candidate_ranking[0][0]
            boundary = stage in ("coarse", "coarse_formal_audit") and winner in (min(r["lambda"] for r in stage_rows), max(r["lambda"] for r in stage_rows))
            if not boundary or extension == 3:
                break
            extension += 1
            candidates = [winner/10 if winner == min(r["lambda"] for r in stage_rows) else winner*10]
        write_json(out/f"{stage}_ranking.json", candidate_ranking)
        if stage == "coarse":
            coarse_candidates = np.array(sorted({r["lambda"] for r in stage_rows}))
        if boundary:
            write_json(out/"tuning_report.json", {"status": "search_insufficient", "stage": stage})
            return
    write_json(out/"tuning_report.json", {"status": "validation_selected_not_frozen", "lambda": candidate_ranking[0][0],
                                           "ranking": candidate_ranking, "tolerance": args.tolerance,
                                           "note": "accuracy/ranking audit and the complete formal gate are still required"})


def _profile_job(spec):
    # Worker receives data paths and one independent task, never warm state.
    reader = DatasetReader(spec["data"])
    op = dataset_operator(reader, spec["gamma"], spec["cache"])
    if spec.get("scene_type") == "synthetic":
        from .points import dirty_fwhm
        truth, _, _ = source_scene(op, sources=spec["task_index"] % 3, distance=.8*dirty_fwhm(op, 15),
                                    bounds=(reader.metadata["bt_min"], reader.metadata["bt_max"]))
        y = experiment_noise(op.reference(truth), physical_sigma(reader), 2, 100+spec["task_index"], 0)
    else:
        y = reader.load(spec["scene"], spec["gamma"])["y"]
    p = Problem(op, y, physical_sigma(reader),
                (reader.metadata["bt_min"], reader.metadata["bt_max"]), spec["scene"])
    start = time.perf_counter()
    result = solve(p, spec["method"], spec["lambda"], tolerance=spec["tolerance"], max_calls=spec["max_calls"])
    target = Path(spec["output"]); target.mkdir()
    record = persist_result(target, result, p, spec)
    record["seconds_job_with_write"] = time.perf_counter()-start
    return record


def _external_profile_job(spec):
    # File-based coordination also works in Windows sandboxes that deny
    # multiprocessing named pipes. Numerical work still runs in CPU processes.
    spec_path = Path(spec["output"]+"_spec.json")
    result_path = Path(spec["output"]+"_worker_result.json")
    write_json(spec_path, {**spec, "worker_result": str(result_path)})
    with spec_path.with_suffix(".log").open("w", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/"experiment.py"),
                                    "worker", "--spec", str(spec_path)], stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    if completed.returncode:
        return {"converged": False, "status": "worker_failed", "rss_bytes": 0,
                "error_log": str(spec_path.with_suffix(".log")), "exit_code": completed.returncode}
    return load_json(result_path)


def worker_command(args):
    spec = load_json(args.spec)
    write_json(spec["worker_result"], _profile_job(spec))


def profile_command(args):
    reader = DatasetReader(args.data)
    manifest = verified_manifest(reader, args.manifest)
    out = new_output(args.out)
    write_json(out/"environment.json", environment())
    cache = out/"geometry_cache"
    # Parent creates files, workers only open immutable mmap arrays.
    before = time.perf_counter()
    dataset_operator(reader, args.gamma, cache).norm_squared_bound()
    cache_seconds = time.perf_counter()-before
    results = []
    for workers in (1, 2, 4, 8):
        target = out/f"workers_{workers}"; target.mkdir()
        specs = [{"data": str(Path(args.data).resolve()), "scene": task["scene"], "gamma": args.gamma,
                  "method": args.method, "lambda": args.lam, "tolerance": args.tolerance,
                  "scene_type": args.scene_type, "task_index": i,
                  "max_calls": args.max_calls, "cache": str(cache.resolve()),
                  "output": str((target/f"task_{i:03d}").resolve())}
                 for i, task in enumerate(manifest["precision_natural"][:args.tasks])]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_external_profile_job, specs))
        elapsed = time.perf_counter()-start
        # Sum per-worker maximum plus parent is conservative for current jobs.
        rss_upper = resident_memory()+sum(sorted((r["rss_bytes"] for r in records), reverse=True)[:workers])
        completed = sum(r["converged"] for r in records)
        entry = {"workers": workers, "blas_threads_per_worker": 1, "tasks": len(records),
                 "method": args.method, "gamma": args.gamma, "phase": args.budget_phase,
                 "scene_type": args.scene_type, "batch": 1, "tolerance": args.tolerance,
                 "measurement_stage": "validation_resource_pilot", "unit": "tasks",
                 "converged": completed == len(records),
                 "completed_converged": completed, "seconds_end_to_end": elapsed,
                 "all_task_throughput": len(records)/elapsed,
                 "completed_throughput": completed/elapsed,
                 "rss_conservative_bytes": rss_upper, "within_memory_budget": rss_upper <= .7*physical_memory(),
                 "records": records}
        results.append(entry)
        write_json(out/"progress.json", results)
        print(json.dumps({k: entry[k] for k in ("workers", "seconds_end_to_end", "completed_converged", "within_memory_budget")}), flush=True)
    eligible = [r for r in results if r["within_memory_budget"] and r["completed_converged"] == r["tasks"]]
    chosen = max(eligible, key=lambda r: r["completed_throughput"])["workers"] if eligible else None
    write_json(out/"resource_report.json", {"passed": False, "cpu_best_workers": chosen,
               "budget_records": eligible,
               "cache_seconds": cache_seconds, "cpu_groups": results,
               "wall_budget_complete": False, "network": "unmeasured: checkpoint/adapter not provided",
               "peak_total_rss_bytes": max(r["rss_conservative_bytes"] for r in results),
               "note": "one method/gamma diagnostic does not complete the required resource pilot"})


def budget_command(args):
    records = []
    for path in args.records:
        loaded = load_json(path)
        records.extend(loaded if isinstance(loaded, list) else loaded.get("cpu_groups", loaded.get("budget_records", loaded.get("records", [loaded]))))
    workload = load_json(args.workload)
    report = wall_budget(records, workload)
    report["applied_workload"] = workload
    out = new_output(args.out)
    write_json(out/"wall_budget.json", report)
    lines = ["# SAIR 墙钟预算", "", "当前正式执行为CPU单worker、网络单图调用；仅使用相同执行方式的已收敛实测任务。并行及batch>1结果仅供诊断。未测分组不代入假定速度。", "",
             "|方法|孔径|阶段|单位|任务数|有效实测数|估计秒数|", "|---|---|---|---|---:|---:|---:|"]
    for g in report["groups"]:
        seconds = f'{g["estimated_seconds"]:.1f}' if g["estimated_seconds"] is not None else "未测"
        lines.append(f'|{g["method"]}|{g["gamma"]}|{g["phase"]}|{g["unit"]}|{g["count"]}|{g["completed_tasks"]}|{seconds}|')
    lines += ["", "总墙钟："+(f'{report["total_seconds"]:.1f} 秒' if report["complete"] else "未测完整，暂不外推。"),
              "", "p95 成本情景不是总时长置信上界。并发吞吐、预计算、启动、投影、匹配滤波、精度审计、训练与验证需单列。"]
    (out/"WALL_CLOCK_BUDGET.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def gate_command(args):
    report = formal_gate(load_json(args.accuracy), load_json(args.resources), load_json(args.frozen))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["allowed"]:
        raise SystemExit(2)


def points_command(args):
    from .point_runs import run_points, tune_l1, network_adapter
    reader = DatasetReader(args.data)
    verified_manifest(reader, args.manifest)
    if args.phase == "formal":
        if not all((args.accuracy, args.resources, args.frozen)):
            raise ValueError("formal source tests require precision/resource reports and a frozen configuration")
        frozen = load_json(args.frozen)
        gate = formal_gate(load_json(args.accuracy), load_json(args.resources), frozen)
        if not gate["allowed"]:
            raise ValueError("formal gate blocked: "+", ".join(gate["blocking_reasons"]))
        if args.limit:
            raise ValueError("formal source ensembles cannot be silently reduced with --limit")
        manifest = verified_manifest(reader, args.manifest)
        if frozen.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("frozen configuration belongs to another manifest")
        args.lam = frozen["lambdas"]["l1"]["g100"]
        args.tolerance = frozen["tolerances"]["l1"]
        args.max_calls = frozen["max_calls"]
    out = new_output(args.out)
    write_json(out/"environment.json", environment())
    thresholds = frozen["thresholds"] if args.phase == "formal" else (load_json(args.thresholds) if args.thresholds else {})
    network = network_adapter(args.network, args.checkpoint) if args.method == "network" else None
    if args.phase == "formal" and network is not None and network.checkpoint_sha256 != frozen["network_checkpoint_sha256"]:
        raise ValueError("network checkpoint differs from the frozen shared model")
    op = dataset_operator(reader, 1.)
    if args.phase == "tuning":
        if args.method != "l1":
            raise ValueError("point parameter search applies to L1")
        tune_l1(reader, op, out, args.tolerance, args.max_calls, thresholds)
    else:
        run_points(reader, op, out, args.phase, args.method, args.lam, args.tolerance,
                   args.max_calls, thresholds, network, args.limit)


def network_profile_command(args):
    from .point_runs import network_adapter
    from .design import network_packet
    reader = DatasetReader(args.data)
    manifest = verified_manifest(reader, args.manifest)
    out = new_output(args.out)
    start = time.perf_counter()
    network = network_adapter(args.network, args.checkpoint)
    startup = time.perf_counter()-start
    backend = network.predict
    if not all(hasattr(backend, k) for k in ("predict_batch", "synchronize")):
        raise ValueError("GPU timing adapter must expose predict_batch(packets) and synchronize()")
    records, budget_records = [], []
    prepared = []
    precompute_start = time.perf_counter()
    if args.scene_type == "synthetic":
        from .point_runs import point_packet
        from .points import dirty_fwhm
        op = dataset_operator(reader, 1.)
        bounds = reader.metadata["bt_min"], reader.metadata["bt_max"]
        for i in range(8):
            truth, _, _ = source_scene(op, sources=i % 3, distance=.8*dirty_fwhm(op, 15), bounds=bounds)
            y = experiment_noise(op.reference(truth), physical_sigma(reader), 2, 200+i, 0)
            prepared.append(point_packet(op, y, physical_sigma(reader), bounds))
    precompute_seconds = time.perf_counter()-precompute_start
    gammas = (1.,) if prepared else (1., .75, .5, .35)
    for gamma in gammas:
        def packets_for(batch):
            return prepared[:batch] if prepared else [network_packet(reader, s, gamma) for s in manifest["tuning_fine"][:batch]]
        for batch in (1, 4, 8):
            def predict(packets):
                # The formal executor calls predict(packet), not predict_batch([packet]).
                if batch == 1:
                    return np.asarray(backend(packets[0]), dtype=float)[None]
                return np.asarray(backend.predict_batch(packets), dtype=float)
            predict(packets_for(batch)); backend.synchronize()
            timings = []
            for repeat in range(args.repeats):
                backend.synchronize(); start = time.perf_counter()
                packets = packets_for(batch)
                predictions = predict(packets)
                backend.synchronize()
                if predictions.shape != (batch, reader.metadata["n_grid"], reader.metadata["n_grid"]) or not np.isfinite(predictions).all():
                    raise ValueError("invalid GPU batch predictions")
                np.savez(out/f"g{gamma}_batch_{batch}_repeat_{repeat:03d}.npz", temperature=predictions)
                elapsed = time.perf_counter()-start
                timings.append(elapsed)
                records.append({"batch": batch, "gamma": gamma, "scene_type": args.scene_type,
                                "seconds_end_to_end": elapsed, "images_per_second": batch/elapsed})
            phases = ("point_formal", "calibration") if prepared else (("main", "extra") if gamma == 1 else ("robustness",))
            for phase in phases:
                budget_records.append({"method": "network", "gamma": gamma, "phase": phase,
                        "scene_type": args.scene_type, "tolerance": None, "batch": batch,
                        "execution_api": "single_packet" if batch == 1 else "predict_batch",
                        "converged": True, "unit": "tasks", "workers": 1, "tasks": args.repeats*batch,
                        "seconds_end_to_end": float(sum(timings)),
                        "completed_throughput": args.repeats*batch/sum(timings),
                        "p50_batch_seconds": float(np.median(timings)), "p95_batch_seconds": float(np.quantile(timings, .95)),
                        "measurement_stage": "validation_resource_pilot", "cost_scope": "inference_transfer_and_write"})
    write_json(out/"network_performance.json", {"startup_seconds": startup, "records": records,
                  "budget_records": budget_records, "precompute_seconds": precompute_seconds,
                  "checkpoint_sha256": network.checkpoint_sha256, "processes": 1,
                  "environment": environment()})


def resources_command(args):
    reports = [load_json(p) for p in args.cpu_reports]
    networks = [load_json(p) for p in args.network_report]
    budget = load_json(args.budget)
    coverage = set()
    peak = 0
    for report in reports:
        serial = [r for r in report.get("cpu_groups", []) if r["workers"] == 1
                  and r.get("converged") and r.get("within_memory_budget")]
        for group in serial:
            coverage.add((group["method"], group["gamma"], group["scene_type"]))
            peak = max(peak, group["rss_conservative_bytes"])
    expected = {(m, g, "natural") for m in ("tikhonov", "tv") for g in (1., .75, .5, .35)} | {("l1", 1., "synthetic")}
    gpu_records = [r for network in networks for r in network.get("records", [])]
    expected_gpu = {(g, "natural") for g in (1., .75, .5, .35)} | {(1., "synthetic")}
    gpu_ok = all(network.get("processes") == 1 for network in networks) and len({n["checkpoint_sha256"] for n in networks}) == 1 and all(
        sum(r["batch"] == b and r["gamma"] == g and r["scene_type"] == kind for r in gpu_records) >= 30
        for b in (1, 4, 8) for g, kind in expected_gpu)
    execution_ok = budget.get("execution") == FORMAL_EXECUTION
    complete = expected <= coverage and gpu_ok and budget.get("complete") is True and execution_ok and peak <= .7*physical_memory()
    out = new_output(args.out)
    write_json(out/"resource_report.json", {"passed": complete, "wall_budget_complete": budget.get("complete") is True,
               "peak_total_rss_bytes": peak, "cpu_coverage": sorted(coverage), "gpu_complete": gpu_ok,
               "execution": dict(FORMAL_EXECUTION) if execution_ok else None,
               "cpu_reports": args.cpu_reports, "network_report": args.network_report, "budget": args.budget})


def freeze_command(args):
    accuracy, resources = load_json(args.accuracy), load_json(args.resources)
    selection = load_json(args.selection)
    checkpoint = Path(args.checkpoint)
    import hashlib
    frozen = {"status": "frozen", "protocol_version": PROTOCOL_VERSION,
              "execution": dict(FORMAL_EXECUTION), "max_calls": FORMAL_MAX_CALLS,
              "manifest_sha256": load_json(args.manifest)["manifest_sha256"],
              "lambdas": selection["lambdas"], "thresholds": load_json(args.thresholds),
              "network_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "tolerances": {"natural": 1e-4 if accuracy.get("natural_passed") else 1e-5,
                             "l1": 1e-4 if accuracy.get("l1_passed") else 1e-5}}
    gate = formal_gate(accuracy, resources, frozen)
    if not gate["allowed"]:
        frozen["status"] = "draft_blocked"
    out = new_output(args.out)
    write_json(out/"frozen.json", frozen); write_json(out/"gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def natural_command(args):
    from .design import network_packet
    from .point_runs import network_adapter
    from .evaluation import NumericalProjector, observation_metrics
    reader = DatasetReader(args.data)
    manifest = verified_manifest(reader, args.manifest)
    frozen = load_json(args.frozen)
    gate = formal_gate(load_json(args.accuracy), load_json(args.resources), frozen)
    if not gate["allowed"]:
        raise ValueError("formal gate blocked: "+", ".join(gate["blocking_reasons"]))
    if frozen.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("frozen configuration belongs to a different manifest")
    methods = args.methods.split(",")
    if not set(methods) <= {"tikhonov", "tv", "network"} or not methods:
        raise ValueError("unknown natural-scene method")
    if args.phase == "extra" and methods != ["network"]:
        raise ValueError("full-test supplementation is network-only")
    network = network_adapter(args.network, args.checkpoint) if "network" in methods else None
    if network is not None and network.checkpoint_sha256 != frozen["network_checkpoint_sha256"]:
        raise ValueError("network differs from the frozen shared checkpoint")
    out = new_output(args.out)
    write_json(out/"frozen.json", frozen); write_json(out/"environment.json", environment())
    scenes = manifest[{"main": "main", "robustness": "robustness", "extra": "network_extra"}[args.phase]]
    gammas = (.75, .5, .35) if args.phase == "robustness" else (1.,)
    records = []
    for gamma in gammas:
        setup_start = time.perf_counter()
        op = dataset_operator(reader, gamma)
        projector = NumericalProjector(op)
        write_json(out/f"projection_{gamma}.json", {"seconds": time.perf_counter()-setup_start,
                   "cutoff": projector.cutoff, "rank": projector.rank, "operator": op.identity})
        for i, scene in enumerate(scenes):
            obs = reader.load(scene, gamma)
            # Truth is acquired for evaluation only, never passed to a method.
            truth = reader.truth(scene)
            for method in methods:
                target = out/f"g{gamma}_{i:04d}_{method}"; target.mkdir()
                start = time.perf_counter()
                try:
                    metadata = {"method": method, "scene": scene, "gamma": gamma, "phase": args.phase,
                                "scene_type": "natural", "batch": 1, "workers": 1, "split": "test"}
                    if method == "network":
                        prediction = network(network_packet(reader, scene, gamma))
                        np.savez(target/"prediction.npz", temperature=prediction)
                        record = {**metadata, "converged": True, "status": "inference_complete", **image_metrics(prediction, truth)}
                        record.update(observation_metrics(op, prediction, obs["y"], physical_sigma(reader), op(truth)))
                    else:
                        from sair_dataset.array import gamma_tag
                        lam = frozen["lambdas"][method][gamma_tag(gamma)]
                        tol = frozen["tolerances"]["natural"]
                        problem = Problem(op, obs["y"], physical_sigma(reader),
                                          (reader.metadata["bt_min"], reader.metadata["bt_max"]), f"{scene}/{gamma}/0")
                        result = solve(problem, method, lam, tolerance=tol, max_calls=frozen["max_calls"])
                        prediction = problem.temperature(result.z)
                        record = persist_result(target, result, problem, {**metadata, "lambda": lam, "tolerance": tol}, truth, op(truth))
                    record.update(projector.metrics(prediction-truth))
                    record["seconds_end_to_end"] = time.perf_counter()-start
                    write_json(target/"result.json", record)
                    records.append(record)
                except Exception as exc:
                    failure = {"method": method, "scene": scene, "gamma": gamma, "status": "failed",
                               "converged": False, "error": repr(exc), "traceback": traceback.format_exc()}
                    write_json(target/"failure.json", failure); records.append(failure)
            if i % 10 == 0:
                write_json(out/"progress.json", {"completed": len(records), "records": records})
    write_json(out/"comparison.json", {"phase": args.phase, "records": records, "execution": dict(FORMAL_EXECUTION),
               "failed_or_unconverged": sum(not r["converged"] for r in records)})


def main():
    parser = argparse.ArgumentParser(description="SAIR 对比实验：固定清单、求解、精度审计、调参、性能与预算")
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p, manifest=False):
        p.add_argument("--data", default="dataset_sr")
        p.add_argument("--out", required=True)
        if manifest:
            p.add_argument("--manifest", required=True)
    def solver_options(p, methods=("tikhonov", "tv")):
        p.add_argument("--method", choices=methods, default="tikhonov")
        p.add_argument("--gamma", type=float, default=1.)
        p.add_argument("--lambda", dest="lam", type=float, default=1.)
        p.add_argument("--tolerance", type=float, default=1e-5)
        p.add_argument("--max-calls", type=int, default=10000)
    p = sub.add_parser("plan"); common(p); p.set_defaults(func=plan_command)
    p = sub.add_parser("solve"); common(p); solver_options(p)
    p.add_argument("--scene", required=True); p.add_argument("--repeat", type=int, default=0); p.set_defaults(func=solve_command)
    p = sub.add_parser("precision"); common(p, True)
    p.add_argument("--natural-count", type=int, default=20); p.add_argument("--synthetic-count", type=int, default=30)
    p.add_argument("--lambda", dest="lam", type=float, default=1.)
    p.add_argument("--max-calls", type=int, default=40000)
    p.add_argument("--thresholds"); p.add_argument("--selection"); p.add_argument("--stop-on-reference-failure", action="store_true")
    p.set_defaults(func=precision_command)
    p = sub.add_parser("tune"); common(p, True); solver_options(p)
    p.add_argument("--final-candidates", type=int, default=2); p.set_defaults(func=tune_command)
    p = sub.add_parser("profile"); common(p, True); solver_options(p, ("tikhonov", "tv", "l1"))
    p.add_argument("--tasks", type=int, default=8); p.set_defaults(func=profile_command)
    p.add_argument("--budget-phase", default="main", choices=("main", "robustness", "point_formal", "calibration"))
    p.add_argument("--scene-type", default="natural", choices=("natural", "synthetic"))
    p = sub.add_parser("budget"); p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--workload", required=True); p.add_argument("--out", required=True); p.set_defaults(func=budget_command)
    p = sub.add_parser("gate"); p.add_argument("--accuracy", required=True); p.add_argument("--resources", required=True)
    p.add_argument("--frozen", required=True); p.set_defaults(func=gate_command)
    p = sub.add_parser("worker", help=argparse.SUPPRESS); p.add_argument("--spec", required=True); p.set_defaults(func=worker_command)
    p = sub.add_parser("points"); common(p, True)
    p.add_argument("--phase", choices=("tuning", "calibration", "formal"), required=True)
    p.add_argument("--method", choices=("l1", "clean", "network", "matched_filter"), required=True)
    p.add_argument("--lambda", dest="lam", type=float, default=1.)
    p.add_argument("--tolerance", type=float, default=1e-5); p.add_argument("--max-calls", type=int, default=10000)
    p.add_argument("--thresholds"); p.add_argument("--limit", type=int)
    p.add_argument("--network"); p.add_argument("--checkpoint")
    p.add_argument("--accuracy"); p.add_argument("--resources"); p.add_argument("--frozen"); p.set_defaults(func=points_command)
    p = sub.add_parser("profile-network"); common(p, True)
    p.add_argument("--network", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--repeats", type=int, default=30); p.set_defaults(func=network_profile_command)
    p.add_argument("--scene-type", choices=("natural", "synthetic"), default="natural")
    p = sub.add_parser("compare"); common(p, True)
    p.add_argument("--phase", choices=("main", "robustness", "extra"), required=True)
    p.add_argument("--methods", default="tikhonov,tv,network")
    p.add_argument("--network"); p.add_argument("--checkpoint")
    p.add_argument("--accuracy", required=True); p.add_argument("--resources", required=True); p.add_argument("--frozen", required=True)
    p.set_defaults(func=natural_command)
    p = sub.add_parser("resources"); p.add_argument("--cpu-reports", nargs="+", required=True)
    p.add_argument("--network-report", nargs="+", required=True); p.add_argument("--budget", required=True)
    p.add_argument("--out", required=True); p.set_defaults(func=resources_command)
    p = sub.add_parser("freeze"); p.add_argument("--accuracy", required=True); p.add_argument("--resources", required=True)
    p.add_argument("--selection", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--thresholds", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True); p.set_defaults(func=freeze_command)
    args = parser.parse_args()
    if hasattr(args, "tasks") and not 1 <= args.tasks <= 20:
        parser.error("--tasks must be in [1,20]")
    args.func(args)
