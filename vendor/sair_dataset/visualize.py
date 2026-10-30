"""Brightness temperature and complex visibility plots, without gridding y.

Matplotlib is optional and is imported only when rendering. A dense reference
evaluates the exact GT forward model on a regular u/v grid; it is neither an
interpolation of measured samples nor a reconstruction from measurements.
"""

import argparse
from pathlib import Path

import numpy as np

from .array import all_baselines, gamma_tag, y_array
from .operator import ForwardOperator, pixel_weights
from .reader import DatasetReader


def dense_visibility(truth, theta, uv_axis):
    """Return complex128 V[u_index,v_index] via separable model evaluation.

    No full measurement matrix is created. At (u,v)=(0,0) the result equals
    the weighted image mean. Plot V.T for horizontal u and vertical v.
    """
    truth = np.asarray(truth, dtype=np.float64)
    uv_axis = np.asarray(uv_axis, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1] or not np.isfinite(truth).all():
        raise ValueError("truth must be a finite square image")
    if uv_axis.ndim != 1 or not len(uv_axis) or not np.isfinite(uv_axis).all():
        raise ValueError("uv_axis must be a finite nonempty vector")
    n = len(truth)
    sky_axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2) * theta / n
    phase = np.exp(-2j * np.pi * np.outer(uv_axis, sky_axis))
    return (phase @ (truth * pixel_weights(n, theta))) @ phase.T


def chinese_font_family(font_manager):
    """Use an installed CJK font instead of silently drawing missing glyphs."""
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "SimHei",
                   "PingFang SC", "Heiti SC", "WenQuanYi Zen Hei", "DengXian", "SimSun"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        return family
    raise ValueError("未找到中文字体，请安装微软雅黑、思源黑体或 Noto Sans CJK SC 后重新运行")


def hermitian_extension(u, v, values=None):
    """Display both directions; mirrored values are conjugates, not new data.

    Do not feed this display extension to a likelihood as independent rows:
    the mirrored noise is the conjugate of the original noise.
    """
    u, v = np.asarray(u), np.asarray(v)
    if u.ndim != 1 or u.shape != v.shape:
        raise ValueError("u/v must be matching one-dimensional vectors")
    if values is not None:
        values = np.asarray(values)
        if values.shape != u.shape:
            raise ValueError("visibility shape must match u/v")
        values = np.concatenate((values, values.conj()))
    return np.concatenate((u, -u)), np.concatenate((v, -v)), values


def render_scene(reader, scene, gamma=1.0, repeat=0, output=None, grid_size=129, dpi=160, show=False,
                 uv_view="hermitian"):
    """Save one eight-panel figure and optionally open a Matplotlib window."""
    if not isinstance(reader, DatasetReader):
        reader = DatasetReader(reader)
    if grid_size < 17 or grid_size > 1025 or grid_size % 2 != 1:
        raise ValueError("grid_size must be odd and between 17 and 1025 (includes DC)")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if uv_view not in {"hermitian", "stored"}:
        raise ValueError("uv_view must be hermitian or stored")
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.colors import LogNorm, Normalize
        from matplotlib.patches import Circle
    except ImportError as error:
        raise ValueError("Plotting requires: python -m pip install -r requirements-visualization.txt") from error

    measurement = reader.load(scene, gamma, repeat)
    truth = reader.truth(scene)
    meta, geometry = reader.metadata, reader.geometry
    config = meta["config"]
    positions = y_array(config["n_arm"], config["spacing_lambda"], config["freq"])
    full_u, full_v = all_baselines(positions, config["freq"])
    if not (np.array_equal(full_u.astype(np.float32), geometry["u"]) and
            np.array_equal(full_v.astype(np.float32), geometry["v"])):
        raise ValueError("stored geometry does not match float64 array definition")
    selected = geometry[f"indices_{gamma_tag(gamma)}"]
    clean_full = ForwardOperator(meta["n_grid"], meta["theta"], full_u, full_v)(truth)
    clean = clean_full[selected]
    u, v = full_u[selected], full_v[selected]
    y = measurement["y"]
    if uv_view == "hermitian":
        display_full_u, display_full_v, _ = hermitian_extension(full_u, full_v)
        display_u, display_v, display_y = hermitian_extension(u, v, y)
    else:
        display_full_u, display_full_v = full_u, full_v
        display_u, display_v, display_y = u, v, y
    radius = np.hypot(u, v)
    r_max = float(np.hypot(full_u, full_v).max())
    uv_axis = np.linspace(-r_max, r_max, grid_size)
    dense = dense_visibility(truth, meta["theta"], uv_axis)
    sigma = float(measurement["sigma"])
    vmax = max(float(np.abs(dense).max()), float(np.abs(y).max()), 1e-12)
    vmin = min(max(vmax * 1e-6, sigma * .1, 1e-12), vmax * .1)
    amplitude_norm = LogNorm(vmin=vmin, vmax=vmax)
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    spacing = uv_axis[1] - uv_axis[0]
    uv_extent = [uv_axis[0]-spacing/2, uv_axis[-1]+spacing/2,
                 uv_axis[0]-spacing/2, uv_axis[-1]+spacing/2]
    styles = {"font.family": [chinese_font_family(font_manager), "DejaVu Sans"],
              "font.size": 10, "axes.titlesize": 11, "axes.unicode_minus": False,
              "axes.labelsize": 10, "legend.fontsize": 8, "figure.facecolor": "white",
              "svg.fonttype": "path"}
    with plt.rc_context(styles):
        figure, axes = plt.subplots(2, 4, figsize=(21, 10), layout="constrained")
        temperature, coverage, true_amplitude, true_phase = axes[0]
        measured_amplitude, measured_phase, radial, components = axes[1]
        split_name = {"train": "训练集", "val": "验证集", "test": "测试集"}.get(
            measurement["split"], measurement["split"])
        figure.suptitle(
            f"场景：{scene}  |  {split_name}  |  γ={gamma:.2f}  |  "
            f"基线数={measurement['m']}  |  重复编号={repeat}  |  σ={sigma:.5f} K",
            fontsize=15, fontweight="bold")
        half = meta["theta"] / 2
        image = temperature.imshow(truth, origin="upper", extent=[-half, half, half, -half],
                                   cmap="inferno", vmin=meta["bt_min"], vmax=meta["bt_max"],
                                   interpolation="nearest")
        temperature.set(title=f"亮温真值（{meta['n_grid']} × {meta['n_grid']}）",
                        xlabel="η / rad（图像列）", ylabel="ξ / rad（图像行）")
        figure.colorbar(image, ax=temperature, label="亮温 / K", shrink=.8)
        coverage.scatter(display_full_u, display_full_v, s=8, c="#d5dae2",
                         label=f"全阵列（{len(full_u)} 对）", linewidths=0)
        coverage.scatter(display_u, display_v, s=9, c="#2166ac", alpha=.75,
                         label=f"保留基线（{len(u)} 对）", linewidths=0, rasterized=True)
        coverage.legend(loc="upper right")
        coverage.set_title("等效基线覆盖（含共轭方向）" if uv_view == "hermitian" else "基线覆盖（原始存盘方向）")

        true_abs = true_amplitude.imshow(np.abs(dense).T, origin="lower", extent=uv_extent,
                                        cmap="magma", norm=amplitude_norm, interpolation="nearest")
        true_arg = true_phase.imshow(np.angle(dense).T, origin="lower", extent=uv_extent,
                                    cmap="twilight", norm=phase_norm, interpolation="nearest")
        true_amplitude.set_title("无噪可见度幅度 |V(u,v)|")
        true_phase.set_title("无噪可见度相位 arg V(u,v)")
        observed_abs = measured_amplitude.scatter(display_u, display_v, c=np.abs(display_y), s=10, cmap="magma",
                                                 norm=amplitude_norm, linewidths=0, rasterized=True)
        observed_arg = measured_phase.scatter(display_u, display_v, c=np.angle(display_y), s=10, cmap="twilight",
                                              norm=phase_norm, linewidths=0, rasterized=True)
        measured_amplitude.set_title("带噪幅度 |y|（含共轭镜像）" if uv_view == "hermitian" else "带噪幅度 |y|（存盘方向）")
        measured_phase.set_title("带噪相位 arg y（含共轭镜像）" if uv_view == "hermitian" else "带噪相位 arg y（存盘方向）")
        figure.colorbar(true_abs, ax=true_amplitude, label="K（对数色标）", shrink=.8)
        figure.colorbar(observed_abs, ax=measured_amplitude, label="K（相同色标）", shrink=.8)
        for image, ax in ((true_arg, true_phase), (observed_arg, measured_phase)):
            colorbar = figure.colorbar(image, ax=ax, label="相位 / rad", shrink=.8,
                                      ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            colorbar.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
        for ax in (coverage, true_amplitude, true_phase, measured_amplitude, measured_phase):
            ax.add_patch(Circle((0,0), gamma*r_max, fill=False, linestyle="--", linewidth=1.,
                                edgecolor="#2ca25f" if ax is coverage else "#18c9b6", alpha=.9))
            ax.set(xlabel="u（波长）", ylabel="v（波长）",
                   xlim=(-r_max*1.04, r_max*1.04), ylim=(-r_max*1.04, r_max*1.04))
            ax.set_aspect("equal", adjustable="box")

        floor = max(vmax*1e-12, 1e-14)
        radial.scatter(radius, np.maximum(np.abs(clean), floor), s=9, color="#333333",
                       alpha=.55, label="保留基线上的无噪 V", linewidths=0)
        radial.scatter(radius, np.maximum(np.abs(y), floor), s=8, color="#2879b9",
                       alpha=.45, label="带噪测量 y", linewidths=0)
        if sigma > 0:
            radial.axhline(sigma, color="#d95f02", linestyle="--", label="噪声均方根 σ")
        radial.set(yscale="log", xlabel="基线半径（波长）", ylabel="幅度 / K",
                   title="可见度幅度随基线半径变化", xlim=(0,r_max*1.04))
        radial.legend(loc="upper right")
        components.scatter(radius, y.real, s=8, color="#2879b9", alpha=.5, linewidths=0, label="实部 Re y")
        components.scatter(radius, y.imag, s=8, color="#d95f02", alpha=.5, linewidths=0, label="虚部 Im y")
        components.set_yscale("symlog", linthresh=max(sigma, vmax*1e-4, 1e-10))
        components.axhline(0, color="#909090", linewidth=.6)
        components.set(xlabel="基线半径（波长）", ylabel="可见度 / K（对称对数）",
                       title="带噪可见度的实部与虚部", xlim=(0,r_max*1.04))
        components.legend(loc="upper right")
        for ax in (radial, components):
            ax.axvline(gamma*r_max, color="#2ca25f", linestyle="--", linewidth=1)
            ax.grid(True, which="major", alpha=.2)
        note = ("±uv 为共轭等效方向，镜像不增加独立测量；径向散点保留原始数据。 "
                if uv_view == "hermitian" else "散点仅显示原始存盘方向。 ")
        figure.supxlabel(note + "密集图由真值前向计算；虚线表示孔径边界；同坐标测量点可能重叠。", fontsize=10)
        destination = Path(output) if output else Path("visualizations") / (scene+f"_{gamma_tag(gamma)}_n{repeat}.png")
        if destination.suffix.lower() not in {".png", ".svg"}:
            raise ValueError("output must end with .png or .svg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, facecolor="white")
        if show:
            plt.show()
        plt.close(figure)
    return destination.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Generated dataset directory")
    parser.add_argument("--scene", help="class/stem; defaults to first matching scene")
    parser.add_argument("--gamma", type=float, default=1., help="Stored aperture ratio (default: 1)")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--all-gammas", action="store_true", help="Render every stored gamma instead of --gamma")
    parser.add_argument("--out", type=Path, default=Path("visualizations"), help="Figure output directory")
    parser.add_argument("--format", choices=["png", "svg"], default="png")
    parser.add_argument("--grid-size", type=int, default=129, help="Odd GT-reference u/v grid size, 17..1025")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--show", action="store_true", help="Also open a Matplotlib window")
    parser.add_argument("--uv-view", choices=["hermitian", "stored"], default="hermitian",
                        help="Show conjugate-completed equivalent u/v coverage (default), or only stored rows")
    parser.add_argument("--list-scenes", action="store_true", help="List scenes without plotting")
    parser.add_argument("--class-name", help="Filter by class")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Filter by split")
    parser.add_argument("--limit", type=int, default=20, help="Max scenes printed by --list-scenes")
    args = parser.parse_args()
    try:
        reader = DatasetReader(args.data)
        scenes = [s for s in reader.metadata["selected_scenes"]
                  if (args.class_name is None or s.split("/")[0] == args.class_name)
                  and (args.split is None or reader.metadata["class_assignment"][s.split("/")[0]] == args.split)]
        if not scenes:
            raise ValueError("no scenes match the selected class/split")
        if args.list_scenes:
            if args.limit < 1:
                raise ValueError("limit must be >= 1")
            for scene in scenes[:args.limit]:
                print(f"{scene}\t{reader.metadata['class_assignment'][scene.split('/')[0]]}")
            print(f"Matched scenes: {len(scenes)}")
            return
        scene = args.scene.replace("\\", "/") if args.scene else scenes[0]
        if scene not in scenes:
            raise ValueError("scene is absent or does not match the class/split filters")
        gammas = [item["gamma"] for item in reader.metadata["plan"]] if args.all_gammas else [args.gamma]
        for gamma in gammas:
            output = args.out / (scene+f"_{gamma_tag(gamma)}_n{args.repeat}.{args.format}")
            path = render_scene(reader, scene, gamma, args.repeat, output, args.grid_size, args.dpi, args.show,
                                uv_view=args.uv_view)
            print(f"Saved: {path}", flush=True)
    except (ValueError, OSError, KeyError, IndexError) as error:
        raise SystemExit(f"Visualization failed: {error}") from error


if __name__ == "__main__":
    main()
