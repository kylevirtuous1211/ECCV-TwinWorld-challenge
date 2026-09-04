"""Qualitative figures for the factsheet.

Everything here is read from the submitted archives and the released dataset,
so a verifier can regenerate the figures without any of our scratch state.

    uv run --with matplotlib --with numpy --with pillow \
        python delivery/factsheet_official/fig/make_qualitative.py \
        --dataset-root <TwinWorld_Datasets> --submissions submission --out delivery/factsheet_official/fig
"""
import argparse, io, sys, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "u1")])
GT_VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")])
# TUM ground truth is Open3D-written: double coordinates, colour, no label.
TUM_GT_VERTEX = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
XYZ_VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

# ground, wall, roof, window, other. Checked with the categorical validator:
# lightness band, chroma floor, CVD separation and contrast all pass on white.
CLASS_NAMES = ("ground", "wall", "roof", "window", "other")
CLASS_COLOURS = ("#b45309", "#2563eb", "#be123c", "#7c3aed", "#0d9488")


def read_ply(stream, dtype=VERTEX):
    """The submission format: binary little-endian xyz + classification."""
    header = b""
    while b"end_header" not in header:
        header += stream.readline()
    count = int(next(l for l in header.split(b"\n")
                     if l.startswith(b"element vertex")).split()[-1])
    return np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)


def from_archive(archive, member, dtype=VERTEX):
    with zipfile.ZipFile(archive) as z:
        with z.open(member) as f:
            return read_ply(io.BufferedReader(f), dtype)


def load_metrics(code_root):
    """Score with the shipped reimplementation rather than a second one written
    here. It crops to the truth's bounding box and voxel-downsamples before
    measuring; a figure that skipped either would not agree with the numbers in
    the text."""
    for candidate in (code_root, Path("release"), Path("code"),
                      Path(__file__).resolve().parents[3] / "release"):
        if candidate and (Path(candidate) / "twinworld" / "metrics.py").exists():
            sys.path.insert(0, str(Path(candidate).resolve()))
            from twinworld import metrics
            return metrics
    raise SystemExit("cannot find twinworld/metrics.py; pass --code-root")


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / ((a - b) ** 2).mean())


def figure_renders(args):
    """Held-out view, ground truth against the mean of three seeds."""
    scenes = [("tum/scene_000", "Data_TUM/scene_000", "TUM scene\\_000"),
              ("tum/scene_002", "Data_TUM/scene_002", "TUM scene\\_002"),
              ("gold_coast/scene_009", "Data_Goldcoast/scene_009", "Gold Coast scene\\_009")]
    archive = args.submissions / "submission_v0.30_bcc019.zip"
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.15))
    with zipfile.ZipFile(archive) as z:
        for col, (scene, released, label) in enumerate(scenes):
            # Score all three held-out views and show the median one. Taking
            # the first member instead would have shown scene_009's worst
            # frame, which reads as a claim about Gold Coast that the numbers
            # do not support.
            members = sorted(n for n in z.namelist()
                             if n.startswith(f"{scene}/rgb/") and n.endswith(".png"))
            gt_dir = args.dataset_root / released / "test" / "images"
            scored = []
            for member in members:
                ours = np.asarray(Image.open(io.BytesIO(z.read(member))).convert("RGB"),
                                  dtype=np.float64)
                gt_file = next(p for p in sorted(gt_dir.iterdir())
                               if p.stem == Path(member).stem)
                truth = np.asarray(Image.open(gt_file).convert("RGB"), dtype=np.float64)
                scored.append((psnr(ours, truth), ours, truth))
            scored.sort(key=lambda t: t[0])
            value, ours, truth = scored[len(scored) // 2]
            mean = float(np.mean([v for v, _, _ in scored]))
            for row, (img, tag) in enumerate(((truth, "ground truth"), (ours, "ours"))):
                ax = axes[row, col]
                ax.imshow(img.astype(np.uint8))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#c9c9c9")
                if col == 0:
                    ax.set_ylabel(tag, fontsize=8)
            axes[0, col].set_title(label.replace("\\_", "_"), fontsize=8)
            axes[1, col].set_xlabel(f"{value:.2f} dB  (scene mean {mean:.2f}, n=3)",
                                    fontsize=7.5)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)
    fig.savefig(args.out / "renders.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)
    print("wrote renders.pdf")


def figure_volume(args):
    """The disclosed representation: same scene, same budget, two arrangements."""
    crops = dict(y0=20.0, half=0.06, x=(26.0, 34.0), z=(-75.6, -70.6))
    panels = [("submission_v0.27_scene006.zip", "labeled surface"),
              ("submission_v0.30_bcc019.zip", "labeled BCC volume")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), sharex=True, sharey=True)
    for ax, (name, title) in zip(axes, panels):
        pts = from_archive(args.submissions / name,
                           "gold_coast/scene_011/3D_point_cloud/point_cloud.ply")
        keep = ((np.abs(pts["y"] - crops["y0"]) < crops["half"])
                & (pts["x"] > crops["x"][0]) & (pts["x"] < crops["x"][1])
                & (pts["z"] > crops["z"][0]) & (pts["z"] < crops["z"][1]))
        sel = pts[keep]
        ax.scatter(sel["x"], sel["z"], s=1.6, c="#1f6feb", linewidths=0, rasterized=True)
        ax.set_title(f"{title}, {len(pts) / 1e6:.2f} M points", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c9c9c9")
    # a 1 m scale bar, since the axes carry no ticks
    bar_x, bar_z = crops["x"][0] + 0.4, crops["z"][0] + 0.35
    axes[0].plot([bar_x, bar_x + 1.0], [bar_z, bar_z], color="#1a1a1a", linewidth=1.4)
    axes[0].text(bar_x + 0.5, bar_z + 0.16, "1 m", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(args.out / "volume.pdf", bbox_inches="tight", pad_inches=0.01)
    print("wrote volume.pdf")


def figure_semantic(args):
    """What the semantic term is actually scoring: our labels against the truth,
    and which truth points we are near enough to label at all."""
    from scipy.spatial import cKDTree

    scene, released = "gold_coast/scene_009", "Data_Goldcoast/scene_009"
    gt_path = args.dataset_root / released / "3d_gt" / "point_cloud.ply"
    with open(gt_path, "rb") as handle:
        truth = read_ply(handle, GT_VERTEX)
    ours = from_archive(args.submissions / "submission_v0.30_bcc019.zip",
                        f"{scene}/3D_point_cloud/point_cloud.ply")

    # The official semantic term is truth-driven: every ground-truth point takes
    # the label of the nearest prediction within 10 cm, and takes none if there
    # is no prediction that close. Reproduce exactly that.
    predicted = np.full(len(truth), 255, dtype=np.uint8)
    distance, index = cKDTree(
        np.stack([ours["x"], ours["y"], ours["z"]], axis=1).astype(np.float64)
    ).query(
        np.stack([truth["x"], truth["y"], truth["z"]], axis=1).astype(np.float64),
        distance_upper_bound=0.10, workers=-1)
    matched = np.isfinite(distance)
    predicted[matched] = ours["c"][index[matched]]

    ious = []
    for label in range(len(CLASS_NAMES)):
        is_truth, is_pred = truth["c"] == label, predicted == label
        union = (is_truth | is_pred).sum()
        ious.append((is_truth & is_pred).sum() / union if union else np.nan)

    rng = np.random.default_rng(0)
    take = rng.choice(len(truth), size=min(700_000, len(truth)), replace=False)
    x, y = truth["x"][take], truth["y"][take]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, labels, title in (
            (axes[0], truth["c"][take], "ground-truth labels"),
            (axes[1], predicted[take], "our labels, transferred")):
        for label, (name, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS)):
            keep = labels == label
            ax.scatter(x[keep], y[keep], s=0.12, c=colour, linewidths=0,
                       rasterized=True, label=name if ax is axes[0] else None)
        ax.set_title(title, fontsize=7.5)

    # Panel 3 as a heat map, not a scatter. Top-down, a vertical wall projects
    # onto a line, so scattering reached against unreached overplots and the
    # colour drawn last wins the pixel regardless of how many points it holds.
    # Binning the fraction reached per cell reports the same quantity honestly.
    cells = 220
    grid = [np.linspace(truth["x"].min(), truth["x"].max(), cells),
            np.linspace(truth["y"].min(), truth["y"].max(), cells)]
    total, _, _ = np.histogram2d(truth["x"], truth["y"], bins=grid)
    reached, _, _ = np.histogram2d(truth["x"][matched], truth["y"][matched], bins=grid)
    with np.errstate(invalid="ignore"):
        fraction = np.where(total > 0, reached / total, np.nan)
    image = axes[2].imshow(np.ma.masked_invalid(fraction).T, origin="lower",
                           extent=(grid[0][0], grid[0][-1], grid[1][0], grid[1][-1]),
                           cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"truth reached within 10 cm ({matched.mean() * 100:.1f}%)",
                      fontsize=7.5)
    bar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.02)
    bar.ax.tick_params(labelsize=6)
    bar.set_label("fraction of truth reached", fontsize=6.5)

    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c9c9c9")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=c)
               for c in CLASS_COLOURS]
    fig.legend(handles, list(CLASS_NAMES), loc="lower center", ncol=5, fontsize=7,
               frameon=False, handletextpad=0.3, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0.045, 1, 1))
    fig.savefig(args.out / "semantic.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)
    print("wrote semantic.pdf   per-class IoU " +
          ", ".join(f"{n} {v:.3f}" for n, v in zip(CLASS_NAMES, ious)) +
          f"   mIoU {np.nanmean(ious):.4f}   coverage {matched.mean():.4f}")


def figure_geometry(args):
    """The geometry term made visible: the truth, our exported cloud, and how
    far each truth point is from the nearest thing we reconstructed."""
    from scipy.spatial import cKDTree
    metrics = load_metrics(args.code_root)

    scene, released = "tum/scene_000", "Data_TUM/scene_000"
    with open(args.dataset_root / released / "3d_gt" / "point_cloud.ply", "rb") as handle:
        truth_raw = read_ply(handle, TUM_GT_VERTEX)
    truth = np.stack([truth_raw["x"], truth_raw["y"], truth_raw["z"]], axis=1).astype(np.float64)

    # The archive thins the four development scenes, which the final phase does
    # not score, to spend the point budget on the seven it does. Drawing that
    # thinned cloud would understate the pipeline, so the panel shows what the
    # pipeline exports and the caption gives both scores.
    native = args.native_root / "tum_geometry" / "seed0" / "tum_scene_000_isect" / "point_cloud.ply"
    with open(native, "rb") as handle:
        exported_raw = read_ply(handle, XYZ_VERTEX)
    exported = np.stack([exported_raw["x"], exported_raw["y"], exported_raw["z"]],
                        axis=1).astype(np.float64)
    shipped_raw = from_archive(args.submissions / "submission_v0.30_bcc019.zip",
                               f"{scene}/3D_point_cloud/point_cloud.ply", XYZ_VERTEX)
    shipped = np.stack([shipped_raw["x"], shipped_raw["y"], shipped_raw["z"]],
                       axis=1).astype(np.float64)

    exported_score = metrics.geometry_fscore(exported, truth)
    shipped_score = metrics.geometry_fscore(shipped, truth)

    # Same preparation the scorer applies, so picture and numbers agree.
    ours = metrics.voxel_downsample(metrics.crop_to_box(exported, truth))
    truth_ds = metrics.voxel_downsample(truth)
    to_ours, _ = cKDTree(ours).query(truth_ds, workers=-1)

    rng = np.random.default_rng(0)
    pick = lambda n: rng.choice(n, size=min(500_000, n), replace=False)
    ti, oi = pick(len(truth_ds)), pick(len(ours))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, xyz, index, title in (
            (axes[0], truth_ds, ti, f"ground truth, {len(truth_ds) / 1e6:.1f} M points"),
            (axes[1], ours, oi, f"our exported cloud, {len(ours) / 1e6:.1f} M points")):
        ax.scatter(xyz[index, 0], xyz[index, 1], s=0.10, c=xyz[index, 2],
                   cmap="cividis", linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=7.5)

    image = axes[2].scatter(truth_ds[ti, 0], truth_ds[ti, 1], s=0.10,
                            c=np.minimum(to_ours[ti], 0.20) * 100, cmap="magma_r",
                            vmin=0, vmax=20, linewidths=0, rasterized=True)
    axes[2].set_title("truth to nearest predicted point", fontsize=7.5)
    bar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.02)
    bar.ax.tick_params(labelsize=6)
    bar.set_label("cm, clipped at 20", fontsize=6.5)

    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c9c9c9")
    fig.tight_layout(pad=0.3, w_pad=0.4)
    fig.savefig(args.out / "geometry.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)

    for name, score in (("exported", exported_score), ("shipped", shipped_score)):
        print(f"geometry.pdf {name:8s} " + "  ".join(
            f"@{t * 100:.0f}cm P {score.precision[t]:.3f} R {score.recall[t]:.3f} "
            f"F {score.per_threshold[t]:.3f}" for t in sorted(score.per_threshold)) +
            f"  mean F {score.fscore:.4f}  n={score.predicted_points:,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--submissions", default=Path("submission"), type=Path)
    p.add_argument("--out", default=Path("delivery/factsheet_official/fig"), type=Path)
    p.add_argument("--code-root", default=None, type=Path,
                   help="directory holding twinworld/metrics.py")
    p.add_argument("--native-root", default=Path("submission_final/upload/native"),
                   type=Path, help="the results bundle's native/ directory")
    p.add_argument("--only", default=None,
                   choices=("renders", "volume", "semantic", "geometry"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "renders"):
        figure_renders(args)
    if args.only in (None, "volume"):
        figure_volume(args)
    if args.only in (None, "semantic"):
        figure_semantic(args)
    if args.only in (None, "geometry"):
        figure_geometry(args)


if __name__ == "__main__":
    main()
