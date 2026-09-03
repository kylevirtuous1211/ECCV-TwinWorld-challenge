#!/usr/bin/env python3
"""Measure how far a scene's ground truth sits from its own camera frame.

Two artefacts the organisers ship should already agree. `train/sparse/0/points3D.ply`
is triangulated from the scene's own images, so it is definitionally in the
camera frame everything else is reconstructed in. `3d_gt/point_cloud.ply` is what
the evaluator scores against. Nothing of ours takes part in this comparison, so
whatever disagreement it finds belongs to the provided data.

On Gold Coast they disagree by 1.1 to 1.4 m horizontally and on TUM they do not,
which is the control this needs: the same search run on a registered scene finds
nothing to gain. That matters far beyond a coverage number. A lateral offset
steps off a wall and merely slides along a roof, which is the exact shape of what
the semantic score does - roof scores, wall does not - so a misregistration here
masquerades as a classifier that cannot see walls.

The search is deliberately crude. It maximises the share of COLMAP points that
land within `--radius` of any ground-truth point, over a grid of translations,
coarse then fine. It is not ICP and should not be: ICP would happily find a
rotation too, and a rotation that improves this statistic on a sparse cloud is
much easier to fit than to justify.

What it does not do is take the argmax, because there is no single best cell.
Sliding a cloud along its own surfaces by less than the match radius costs no
matches at all, so the objective is a plateau about that wide, and the first
cell the loop happens to reach sits at its corner - which puts a fixed error of
up to the radius into every offset, in the same direction, on every scene. The
plateau's centre is the answer, so the plateau is averaged.

    python scripts/frame_offset.py --dataset-root <path> --output <json>

What it cannot do is derive the shift for a scene with no ground truth, which is
both withheld Gold Coast scenes. `dy` agrees across the two measured scenes and
`dx` does not, so there is no constant to extrapolate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.pointcloud import read_gaussian_centres  # noqa: E402

# 15 cm is looser than the geometry metric's 5 cm on purpose. This is asking
# "does the frame line up", not "is the reconstruction accurate", and a radius
# at the metric's threshold would let sparse-cloud noise decide the answer.
MATCH_RADIUS = 0.15

COARSE_STEP = 0.25
COARSE_SPAN = 2.5
FINE_STEP = 0.05
FINE_SPAN = 0.35

# A translation that is smaller than the match radius, along a surface, does not
# cost a single match: the point stays on the same plane and its nearest
# neighbour is still under the radius. So the objective is not a peak, it is a
# plateau roughly the width of the radius, and a plain argmax reports whichever
# corner of it the loop reached first - which is a systematic bias, not noise.
# Averaging the plateau instead reports its centre, which is where the surfaces
# actually agree, and it is the reason the TUM control comes out at zero.
PLATEAU_TOLERANCE = 0.005

DATASETS = {"tum": "Data_TUM", "gold_coast": "Data_Goldcoast"}


def matched_share(tree: cKDTree, points: np.ndarray, offset: np.ndarray,
                  radius: float) -> float:
    """The share of `points`, translated by `offset`, that land near the truth.

    `distance_upper_bound` is not an approximation here - the question is only
    ever "is there a neighbour within the radius", and a bounded query answers
    exactly that while abandoning each hopeless point early. It is what makes a
    grid of several hundred translations against a ten-million-point tree
    affordable, and the search is run several hundred times per scene.
    """
    distance, _ = tree.query(points + offset, distance_upper_bound=radius, workers=-1)
    return float(np.isfinite(distance).mean())


def search(tree: cKDTree, points: np.ndarray, axes: str, radius: float,
           step: float, span: float, centre: np.ndarray,
           tolerance: float) -> tuple[np.ndarray, float]:
    """The centre of the translations, on a grid around `centre`, that match most.

    Returns a continuous offset rather than a grid cell, because the plateau's
    centre generally is not one, and quantising it to `step` would put a 2.5 cm
    error into a shift that gets compared against a 10 cm match radius.
    """
    steps = int(round(span / step))
    grid = np.arange(-steps, steps + 1) * step
    candidates = [grid if axis in axes else np.array([0.0]) for axis in "xyz"]

    offsets, shares = [], []
    for dx in candidates[0]:
        for dy in candidates[1]:
            for dz in candidates[2]:
                offset = centre + np.array([dx, dy, dz])
                offsets.append(offset)
                shares.append(matched_share(tree, points, offset, radius))

    offsets, shares = np.array(offsets), np.array(shares)
    plateau = shares >= shares.max() - tolerance
    best = offsets[plateau].mean(axis=0)
    return best, matched_share(tree, points, best, radius)


def measure(scene_root: Path, axes: str, radius: float,
            tolerance: float) -> dict | None:
    truth_path = scene_root / "3d_gt" / "point_cloud.ply"
    colmap_path = scene_root / "train" / "sparse" / "0" / "points3D.ply"
    if not truth_path.exists() or not colmap_path.exists():
        return None

    truth = read_gaussian_centres(truth_path)
    colmap = read_gaussian_centres(colmap_path)
    tree = cKDTree(truth)

    zero = np.zeros(3)
    as_is = matched_share(tree, colmap, zero, radius)
    coarse, _ = search(tree, colmap, axes, radius, COARSE_STEP, COARSE_SPAN,
                       zero, tolerance)
    offset, shifted = search(tree, colmap, axes, radius, FINE_STEP, FINE_SPAN,
                             coarse, tolerance)

    return {"colmap_points": int(len(colmap)),
            "truth_points": int(len(truth)),
            "as_is": round(as_is, 4),
            "offset": [round(float(v), 4) for v in offset],
            "shifted": round(shifted, 4),
            "horizontal_metres": round(float(np.linalg.norm(offset[:2])), 4),
            "vertical_metres": round(float(offset[2]), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--axes", default="xy", choices=["xy", "xyz"],
                        help="which translation axes to search. 'xy' reproduces the "
                             "documented measurement; 'xyz' also asks whether the "
                             "ground truth sits at the wrong height")
    parser.add_argument("--radius", type=float, default=MATCH_RADIUS)
    parser.add_argument("--tolerance", type=float, default=PLATEAU_TOLERANCE,
                        help="how close to the best match a translation has to come "
                             "to count as part of the plateau that gets averaged")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the offsets shift_clouds.py consumes")
    args = parser.parse_args()

    receipt = {"axes": args.axes, "radius": args.radius,
               "tolerance": args.tolerance, "scenes": {}}
    print(f"{'scene':26s} {'colmap':>8s} {'as-is':>7s} "
          f"{'dx':>7s} {'dy':>7s} {'dz':>7s} {'shifted':>8s} {'offset':>7s}")
    print("-" * 82)

    for dataset, folder in DATASETS.items():
        for scene_root in sorted((args.dataset_root / folder).glob("scene_*")):
            started = time.time()
            row = measure(scene_root, args.axes, args.radius, args.tolerance)
            if row is None:
                print(f"{dataset + '/' + scene_root.name:26s} "
                      f"{'no ground truth, so no offset can be derived':>50s}")
                continue
            row["seconds"] = round(time.time() - started, 1)
            receipt["scenes"][f"{dataset}/{scene_root.name}"] = row
            print(f"{dataset + '/' + scene_root.name:26s} {row['colmap_points']:>8,d} "
                  f"{row['as_is']:>6.1%} {row['offset'][0]:>7.2f} {row['offset'][1]:>7.2f} "
                  f"{row['offset'][2]:>7.2f} {row['shifted']:>7.1%} "
                  f"{row['horizontal_metres']:>6.2f}m", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
