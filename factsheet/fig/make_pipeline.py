"""Figure 1: the pipeline, carrying real data through every stage.

Each tile holds the actual object at that point in the method - a released view,
the dense MVS seed, the fused cloud, the voted cloud, the labelled volume, a
submitted render - so the figure shows what the method does rather than only
naming it.

Authored at the width it is printed at. The first version was drawn 13.6 in wide
and placed at \\textwidth, so every label was rendered at half the size it was
set in; working in inches at the final size is the whole fix.

    uv run --with matplotlib --with numpy --with pillow \
        python delivery/factsheet_official/fig/make_pipeline.py \
        --dataset-root <TwinWorld_Datasets> --runs <EditReadyGS_runs/twinworld>
"""
import argparse, io, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

XYZ = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
XYZC = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "u1")])
XYZRGB = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])

INK, MUTED, ACCENT = "#111318", "#5f6672", "#1f6feb"
BAND, FRAME = "#f4f6f8", "#c8ccd2"
CLASS_COLOURS = ("#b45309", "#2563eb", "#be123c", "#7c3aed", "#0d9488")

FIG_W, FIG_H = 7.0, 3.62
TILE_H, PLOT_H = 1.02, 0.60          # inches
ROW_Y = (2.34, 1.20, 0.06)           # tile bottoms, top row first
HEADER_Y = 3.46
GROUPS = [                            # (left, width) in inches
    (0.02, 1.14), (1.35, 1.14), (2.68, 1.36), (4.23, 1.36), (5.78, 1.20)]


def read_ply(stream, dtype):
    header = b""
    while b"end_header" not in header:
        header += stream.readline()
    count = int(next(l for l in header.split(b"\n")
                     if l.startswith(b"element vertex")).split()[-1])
    return np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)


def load(path, dtype):
    with open(path, "rb") as handle:
        return read_ply(handle, dtype)


def from_archive(archive, member, dtype):
    with zipfile.ZipFile(archive) as z, z.open(member) as f:
        return read_ply(io.BufferedReader(f), dtype)


def thin(points, limit, seed=0):
    if len(points) <= limit:
        return points
    return points[np.random.default_rng(seed).choice(len(points), limit, replace=False)]


def fx(inches):
    return inches / FIG_W


def fy(inches):
    return inches / FIG_H


def band(fig, left, width, label):
    """One tinted stage column, with its name above it. Grouping the stages this
    way replaces the per-tile plates, which boxed every caption twice."""
    fig.patches.append(FancyBboxPatch(
        (fx(left - 0.05), fy(-0.02)), fx(width + 0.10), fy(3.36),
        boxstyle="round,pad=0,rounding_size=0.010", transform=fig.transFigure,
        linewidth=0, facecolor=BAND, zorder=-6))
    fig.text(fx(left + width / 2), fy(HEADER_Y), label, ha="center", va="center",
             fontsize=7.0, fontweight="bold", color=MUTED)


def tile(fig, left, bottom, width, title, subtitle=None, gain=None):
    plot = fig.add_axes([fx(left + 0.03), fy(bottom + TILE_H - PLOT_H - 0.02),
                         fx(width - 0.06), fy(PLOT_H)], zorder=2)
    plot.set_facecolor("#ffffff")
    plot.set_xticks([]); plot.set_yticks([])
    for spine in plot.spines.values():
        spine.set_edgecolor(FRAME); spine.set_linewidth(0.6)
    centre = fx(left + width / 2)
    fig.text(centre, fy(bottom + 0.28), title, ha="center", va="center",
             fontsize=6.6, fontweight="bold", color=INK)
    if subtitle:
        fig.text(centre, fy(bottom + 0.155), subtitle, ha="center", va="center",
                 fontsize=5.8, color=MUTED)
    if gain:
        fig.text(centre, fy(bottom + 0.035), gain, ha="center", va="center",
                 fontsize=6.2, fontweight="bold", color=ACCENT)
    return plot


def fill_view(ax, x, y):
    """`adjustable="datalim"` widens the view to satisfy equal aspect; the
    default shrinks the axes inside its tile and leaves white margins."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())


def photo(ax, array, width):
    """Centre-crop to the frame's aspect so the tile is filled. Left alone, a
    4:3 photo in a 2:1 frame sits between two white bars."""
    target = (width - 0.06) / PLOT_H
    h, w = array.shape[:2]
    if w / h > target:
        keep = int(round(h * target))
        start = (w - keep) // 2
        array = array[:, start:start + keep]
    else:
        keep = int(round(w / target))
        start = (h - keep) // 2
        array = array[start:start + keep, :]
    ax.imshow(array)


def cloud(ax, points, colour=None, cmap="cividis", size=0.25):
    x, y = points["x"], points["y"]
    if colour is None:
        ax.scatter(x, y, s=size, c=points["z"], cmap=cmap, linewidths=0, rasterized=True)
    else:
        ax.scatter(x, y, s=size, c=colour, linewidths=0, rasterized=True)
    fill_view(ax, x, y)


def arrow(fig, start, end, curve=0.0):
    fig.patches.append(FancyArrowPatch(
        (fx(start[0]), fy(start[1])), (fx(end[0]), fy(end[1])),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6.5,
        linewidth=0.8, color="#9aa1a9", shrinkA=0.5, shrinkB=0.5,
        connectionstyle=f"arc3,rad={curve}", zorder=-4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--runs", default=Path("/home/intern_2603056/EditReadyGS_runs/twinworld"),
                   type=Path)
    p.add_argument("--submissions", default=Path("submission"), type=Path)
    p.add_argument("--native-root", default=Path("submission_final/upload/native"), type=Path)
    p.add_argument("--out", default=Path("delivery/factsheet_official/fig"), type=Path)
    args = p.parse_args()
    archive = args.submissions / "submission_v0.30_bcc019.zip"
    tum = args.dataset_root / "Data_TUM" / "scene_000"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    for (left, width), label in zip(GROUPS, (
            "input", "initialisation", "three families", "fusion", "submission")):
        band(fig, left, width, label)

    def shrink(image):
        image = image.convert("RGB")
        image.thumbnail((640, 640))
        return np.asarray(image)

    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist()
                      if n.startswith("tum/scene_000/rgb/") and n.endswith(".png"))
        averaged = shrink(Image.open(io.BytesIO(z.read(member))))
        other = next(n for n in z.namelist()
                     if n.startswith("gold_coast/scene_010/rgb/") and n.endswith(".png"))
        elsewhere = shrink(Image.open(io.BytesIO(z.read(other))))
    seed_dir = args.native_root / "renders" / "seed0" / "tum_scene_000" / "rgb"
    single = (shrink(Image.open(sorted(seed_dir.iterdir())[0]))
              if seed_dir.is_dir() else averaged)

    mid = ROW_Y[1]
    g = GROUPS

    ax = tile(fig, g[0][0], mid, g[0][1], "released capture", "12-24 posed views")
    photo(ax, shrink(Image.open(sorted((tum / "train" / "images").iterdir())[0])), g[0][1])

    ax = tile(fig, g[1][0], mid, g[1][1], "dense MVS seed", "poses held fixed", "+0.10 F")
    cloud(ax, thin(load(args.runs / "seed_sweep" / "tum_scene_000_shipped" /
                        "points3D.ply", XYZRGB), 120_000), size=0.8)

    ax = tile(fig, g[2][0], ROW_Y[0], g[2][1], "geometry 2DGS, 5 seeds",
              "ray-disc intersection depth", "+0.083 F")
    cloud(ax, thin(from_archive(archive, "tum/scene_004/3D_point_cloud/point_cloud.ply",
                                XYZ), 120_000))

    gc = from_archive(archive, "gold_coast/scene_009/3D_point_cloud/point_cloud.ply", XYZC)
    ax = tile(fig, g[2][0], mid, g[2][1], "Gold Coast 2DGS, 4 scenes", "colour kept")
    cloud(ax, thin(gc, 120_000))

    ax = tile(fig, g[2][0], ROW_Y[2], g[2][1], "rendering 2DGS, 3 seeds",
              "no distortion term")
    photo(ax, single, g[2][1])

    ax = tile(fig, g[3][0], ROW_Y[0], g[3][1], "agreement filter",
              "2 of 4 within 5 cm", "+0.029 F")
    cloud(ax, thin(from_archive(archive, "tum/scene_005/3D_point_cloud/point_cloud.ply",
                                XYZ), 120_000))

    sample = thin(gc, 120_000)
    ax = tile(fig, g[3][0], mid, g[3][1], "labels, then frame", "random forest")
    cloud(ax, sample, colour=[CLASS_COLOURS[min(int(c), 4)] for c in sample["c"]])

    ax = tile(fig, g[3][0], ROW_Y[2], g[3][1], "mean of three seeds",
              "per-pixel", "+0.026 rend.")
    photo(ax, averaged, g[3][1])

    ax = tile(fig, g[4][0], ROW_Y[0], g[4][1], "13 point clouds", "71.7 M points")
    cloud(ax, thin(from_archive(archive, "tum/scene_007/3D_point_cloud/point_cloud.ply",
                                XYZ), 120_000))

    ax = tile(fig, g[4][0], mid, g[4][1], "occupancy volume",
              "body-centered cubic, $a$ = 0.19 m")
    volume = from_archive(archive, "gold_coast/scene_011/3D_point_cloud/point_cloud.ply", XYZC)
    slab = volume[np.abs(volume["y"] - 20.0) < 0.06]
    slab = slab[(slab["x"] > 26) & (slab["x"] < 34) &
                (slab["z"] > -75.6) & (slab["z"] < -70.6)]
    ax.scatter(slab["x"], slab["z"], s=1.6, c=ACCENT, linewidths=0, rasterized=True)
    fill_view(ax, slab["x"], slab["z"])

    ax = tile(fig, g[4][0], ROW_Y[2], g[4][1], "39 images", "3 held-out views per scene")
    photo(ax, elsewhere, g[4][1])

    plot_mid = lambda bottom: bottom + TILE_H - 0.02 - PLOT_H / 2
    arrow(fig, (g[0][0] + g[0][1], plot_mid(mid)), (g[1][0], plot_mid(mid)))
    for row in ROW_Y:
        curve = 0.0 if row == mid else (0.09 if row > mid else -0.09)
        arrow(fig, (g[1][0] + g[1][1], plot_mid(mid)), (g[2][0], plot_mid(row)), curve)
        arrow(fig, (g[2][0] + g[2][1], plot_mid(row)), (g[3][0], plot_mid(row)))
        arrow(fig, (g[3][0] + g[3][1], plot_mid(row)), (g[4][0], plot_mid(row)))

    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "pipeline.pdf", bbox_inches="tight", pad_inches=0.012, dpi=400)
    print("wrote pipeline.pdf")


if __name__ == "__main__":
    main()
