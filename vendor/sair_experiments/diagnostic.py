"""Four traditional methods on identical measured rows and noise.

This single-scene illustration accepts any existing split, records the original
split, and does not bypass the formal experiment gate. CLEAN components retain their integral units. A fixed,
unit-sum beam restoration is saved separately for display, never relabeled
as a converged box-constrained temperature reconstruction.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter

from sair_dataset.reader import DatasetReader
from .operators import dataset_operator
from .solvers import Problem, solve
from .points import clean, dirty_fwhm, dirty_components
from .point_runs import residual_stop_calibration
from .runtime import environment, new_output, write_json, persist_result, physical_sigma
from .evaluation import image_metrics, observation_metrics


def restore_for_display(op, components, background):
    """Unit-sum Gaussian in integral space, then decode to K; no residual add.

    The beam widths come only from the sampling PSF along array axes. Edge
    cropping is retained. No truth-derived beam, clipping, or peak rescaling.
    """
    widths = np.array([dirty_fwhm(op, 0), dirty_fwhm(op, 90)])*op.n/op.theta
    sigma_pixels = widths/(2*np.sqrt(2*np.log(2)))
    flux_image = gaussian_filter(components, sigma=sigma_pixels, mode="constant", truncate=8.)
    return background+flux_image/op.w, widths


def checked_reuse(directory, problem, scene, gamma, repeat, method, lam, tolerance, max_calls):
    directory = Path(directory)
    record = json.loads((directory/"result.json").read_text(encoding="utf-8"))
    expected = dict(scene=scene, gamma=gamma, repeat=repeat, method=method, **{"lambda": lam, "tolerance": tolerance})
    if any(record.get(k) != v for k, v in expected.items()) or record.get("observation_key") != problem.key:
        raise ValueError("saved result does not match the same observation and solver settings")
    if sum(record["calls"].values()) > max_calls:
        raise ValueError("saved result exceeds this diagnostic's operator budget")
    with np.load(directory/"prediction.npz", allow_pickle=False) as archive:
        prediction = archive["temperature"].copy()
    if prediction.shape != (problem.op.n,)*2 or not np.isfinite(prediction).all():
        raise ValueError("saved prediction is invalid")
    return prediction, record


def render(out, truth, predictions, records, component_image, op, bounds, scene, gamma, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LogNorm
    from sair_dataset.visualize import chinese_font_family
    plt.rcParams.update({"font.sans-serif": [chinese_font_family(font_manager)], "axes.unicode_minus": False})
    methods = ("tikhonov", "tv", "l1", "clean")
    labels = {"tikhonov": "Tikhonov", "tv": "TV", "l1": "非负最小质量 L1", "clean": "CLEAN 固定波束恢复"}
    by_method = {r["method"]: r for r in records}
    fig, axes = plt.subplots(2, 5, figsize=(21, 8), constrained_layout=True)
    im = axes[0, 0].imshow(truth, vmin=bounds[0], vmax=bounds[1], cmap="inferno")
    axes[0, 0].set_title("真值亮温（K）")
    axes[1, 0].axis("off")
    axes[1, 0].text(0, .9, "同一场景、同一可见度、同一噪声\n256×256 网格；补充诊断\n参数未经完整验证选择\n\nCLEAN 恢复图只作展示\n不以恢复图 RMSE 给 CLEAN 排名\n共享网络尚无权重", va="top", fontsize=11)
    for col, method in enumerate(methods, 1):
        record = by_method[method]
        prediction = predictions[method]
        axes[0, col].imshow(prediction, vmin=bounds[0], vmax=bounds[1], cmap="inferno")
        if method == "clean":
            status = "达到停止阈值" if record["status"] == "threshold_reached" else "达到迭代上限"
        else:
            status = "收敛" if record["converged"] else "未收敛，保留输出"
        axes[0, col].set_title(labels[method]+"\n"+status, color="black" if record["converged"] else "red", fontsize=13)
        error_im = axes[1, col].imshow(prediction-truth, vmin=-30., vmax=30., cmap="RdBu_r")
        if method == "clean":
            title = "恢复图误差，仅展示"
        else:
            title = f'RMSE={record["rmse"]:.3f} K'
        axes[1, col].set_title(title, fontsize=12)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=list(axes[0]), label="亮温（K）", shrink=.8)
    fig.colorbar(error_im, ax=list(axes[1, 1:]), label="显示图减真值（K）", shrink=.8)
    fig.suptitle(f"四种传统方法同观测诊断：{scene}，γ={gamma:g}，原始划分={split}", fontsize=17)
    fig.savefig(out/"comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    positive = component_image[component_image > 0]
    if positive.size:
        low = max(float(positive.max())*1e-6, float(positive.min()))
        high = max(float(positive.max()), low*1.01)
        comp = axes[0].imshow(np.ma.masked_less_equal(component_image, 0), norm=LogNorm(low, high), cmap="viridis")
        fig.colorbar(comp, ax=axes[0], label="分量积分源强 c = w·ΔT（归一化积分单位）")
    axes[0].set_title("CLEAN 原始正分量模型（不是像素亮温）")
    restored = axes[1].imshow(predictions["clean"], vmin=bounds[0], vmax=bounds[1], cmap="inferno")
    fig.colorbar(restored, ax=axes[1], label="恢复图亮温（K），仅展示")
    axes[1].set_title("采样波束固定高斯恢复；不添加残差")
    fig.savefig(out/"clean_components.png", dpi=160)
    plt.close(fig)


def run(args):
    reader = DatasetReader(args.data)
    obs = reader.load(args.scene, args.gamma, args.repeat)
    if args.max_calls > 10000 or args.max_calls < 2:
        raise ValueError("use the protocol's <=10000-call formal-candidate budget")
    out = new_output(args.out)
    op = dataset_operator(reader, args.gamma)
    bounds = reader.metadata["bt_min"], reader.metadata["bt_max"]
    sigma = physical_sigma(reader)
    write_json(out/"environment.json", environment())
    observation_hash = hashlib.sha256(op.identity.encode()+obs["y"].tobytes()).hexdigest()
    np.savez(out/"shared_observation.npz", y=obs["y"], u=op.u, v=op.v, sigma_ref=np.array(sigma))
    records, predictions = [], {}
    for method, lam, reuse in (("tikhonov", args.lambda_tikhonov, args.reuse_tikhonov),
                               ("tv", args.lambda_tv, args.reuse_tv), ("l1", args.lambda_l1, None)):
        child = out/method; child.mkdir()
        print(f"{method}: {'verify saved result' if reuse else 'solve same observation'}", flush=True)
        p = Problem(op, obs["y"], sigma, bounds, f"{args.scene}/{args.gamma}/{args.repeat}")
        if reuse:
            prediction, old = checked_reuse(reuse, p, args.scene, args.gamma, args.repeat,
                                           method, lam, args.tolerance, args.max_calls)
            record = {**old, "reused_from": str(Path(reuse).resolve())}
            np.savez(child/"prediction.npz", temperature=prediction)
        else:
            result = solve(p, method, lam, tolerance=args.tolerance, max_calls=args.max_calls)
            prediction = p.temperature(result.z)
            record = persist_result(child, result, p,
                {"method": method, "scene": args.scene, "gamma": args.gamma, "repeat": args.repeat,
                 "lambda": lam, "tolerance": args.tolerance, "observation_key": p.key,
                 "phase": "supplemental_validation_diagnostic"})
        # Truth is used only after reconstruction, for evaluation.
        truth = reader.truth(args.scene)
        record.update(image_metrics(prediction, truth, bounds[1]-bounds[0]))
        record.update(observation_metrics(op, prediction, obs["y"], sigma, op(truth)))
        record["shared_observation_sha256"] = observation_hash
        record["split"] = obs["split"]
        record["formal_comparison"] = False
        write_json(child/"result.json", record)
        records.append(record); predictions[method] = prediction
        write_json(out/"progress.json", records)
        print(json.dumps({k: record.get(k) for k in ("method", "status", "relative_gap", "rmse", "seconds_solver")}), flush=True)
    print("clean: calibrate residual stopping threshold on 650 independent blank samples", flush=True)
    start = time.perf_counter()
    threshold = residual_stop_calibration(op, sigma, bounds)
    calibration_seconds = time.perf_counter()-start
    print(f"clean: solve same observation; calibrated stop={threshold:.8g}", flush=True)
    start = time.perf_counter()
    result = clean(op, obs["y"], background=bounds[0], stop_threshold=threshold, max_iterations=10000)
    elapsed = time.perf_counter()-start
    prediction_vis = obs["y"]-result.residual
    # Independent check of c=w*DeltaT semantics, retaining every original row.
    model_vis = op(bounds[0]+result.components/op.w)
    if not np.allclose(prediction_vis, model_vis, atol=1e-9, rtol=1e-10):
        raise ArithmeticError("CLEAN component units do not replay the measured visibility model")
    restored, fwhm_pixels = restore_for_display(op, result.components, bounds[0])
    clean_y = op(truth)
    record = {"method": "clean", "scene": args.scene, "gamma": args.gamma, "repeat": args.repeat,
        "status": result.status, "converged": result.status == "threshold_reached", "relative_gap": None,
        "rmse": None, "mae": None, "psnr": None, "seconds_solver": elapsed,
        "calibration_seconds": calibration_seconds, "iterations": result.iterations,
        "normalized_noisy_residual": float(np.mean(np.abs(prediction_vis-obs["y"])**2)/sigma**2),
        "clean_visibility_rmse": float(np.sqrt(np.mean(np.abs(prediction_vis-clean_y)**2))),
        "residual_stop_threshold": threshold, "residual_stop_calibration_blank_count": 650,
        "component_flux": float(result.components.sum()), "display_beam_fwhm_pixels": fwhm_pixels.tolist(),
        "display_restored_rmse_not_ranked": image_metrics(restored, truth, bounds[1]-bounds[0])["rmse"],
        "shared_observation_sha256": observation_hash, "split": obs["split"], "formal_comparison": False,
        "meaning": "component model; fixed sampling-beam restoration is for display only"}
    child = out/"clean"; child.mkdir()
    np.savez(child/"components.npz", components=result.components, residual=result.residual,
             display_restored_temperature=restored, display_beam_fwhm_pixels=fwhm_pixels)
    write_json(child/"result.json", record)
    records.append(record); predictions["clean"] = restored
    summary = {"scene": args.scene, "gamma": args.gamma, "repeat": args.repeat, "split": obs["split"],
               "shared_observation_sha256": observation_hash, "formal_comparison": False,
               "parameters_validation_selected": False, "records": records,
               "network": "not trained; no checkpoint supplied", "sigma_ref": sigma,
               "note": "supplemental single scene; original split retained; no method-ranking or superresolution certification"}
    write_json(out/"comparison.json", summary)
    columns = ["method", "split", "status", "lambda", "rmse", "mae", "psnr", "relative_gap", "normalized_noisy_residual", "clean_visibility_rmse", "seconds_solver"]
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    lines = ["# 四种传统方法同观测诊断", "", f"场景 `{args.scene}`，原始划分 `{obs['split']}`，gamma={args.gamma}，repeat={args.repeat}；同一可见度SHA256：`{observation_hash}`。", "",
        "仅单场景补充诊断，保持原始数据划分。参数尚未完整验证选择；未收敛结果不用于方法优劣结论。L1/CLEAN的论文确认性任务仍是点源实验，本例不修改冻结协议。正式参数选择仍只使用val，最终评价使用test；共享网络尚无权重。", "",
        "|方法|状态|lambda|RMSE(K)|归一化含噪残差|无噪可见度RMSE(K)|求解秒数|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        value = f'{r["rmse"]:.4f}' if r["rmse"] is not None else "不排名（分量模型）"
        lam = r.get("lambda", "—")
        lines.append(f'|{r["method"]}|{r["status"]}|{lam}|{value}|{r["normalized_noisy_residual"]:.5f}|{r["clean_visibility_rmse"]:.5f}|{r["seconds_solver"]:.3f}|')
    lines += ["", f"CLEAN停止阈值由650个独立空场校准，校准另耗时{calibration_seconds:.3f}秒。停止阈值不等于完整检测/误分裂认证。", "",
              "CLEAN图像用实际采样波束的两个轴向FWHM作单位和高斯恢复，未加残差、不调恢复波束、不用GT选波束；原始分量与恢复图分别保存。恢复图误差仅供展示，未放入像素RMSE排名。", "",
              "含噪残差更小不自动说明重建更好。Tikhonov/TV若复用既有结果，已核对观测身份、参数、容差和调用上限；时间沿用原求解记录，不是本次统一计时排名。", "",
              "![四种方法](comparison.png)", "", "![CLEAN分量](clean_components.png)"]
    (out/"COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    render(out, truth, predictions, records, result.components, op, bounds, args.scene, args.gamma, obs["split"])
    print(json.dumps({"output": str(out.resolve()), "methods": [r["method"] for r in records], "formal": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="四种传统方法的同观测补充诊断（可选任意现有类别，保留原始划分，不替代正式实验）")
    parser.add_argument("--data", default="dataset_sr")
    parser.add_argument("--scene", default="beach/beach_531")
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--lambda-tikhonov", type=float, default=100.)
    parser.add_argument("--lambda-tv", type=float, default=1.)
    parser.add_argument("--lambda-l1", type=float, default=1.)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--max-calls", type=int, default=10000)
    parser.add_argument("--reuse-tikhonov")
    parser.add_argument("--reuse-tv")
    parser.add_argument("--out", required=True)
    run(parser.parse_args())
