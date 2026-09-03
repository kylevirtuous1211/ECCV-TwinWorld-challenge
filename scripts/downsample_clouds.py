#!/usr/bin/env python3
"""Thin the exported clouds, because density past the metric's voxel is waste.

The geometry metric crops to the ground-truth box, voxel-downsamples at 2 cm and
then measures distances. Everything finer than that voxel is discarded before a
single distance is computed, so a denser cloud cannot score higher - it can only
cost upload bytes and evaluator memory.

Our exports run 15M to 21M points against ground-truth clouds of 4.7M to 11.9M
occupied voxels, and the first complete submission reached the organisers'
scorer, entered Scoring, and died without a log or an exit code. Reducing what
the evaluator has to hold is the cheapest response, and unlike every other lever
its cost is known: the F-score against voxel size was measured over the four dev
scenes and is in HANDOVER.md.

    2 cm  0.3880      3 cm  0.3727      4 cm  0.3620      6 cm  0.3465

Semantics are not handled here. `label_cloud.py --voxel` already produces a
labelled cloud at whatever spacing is wanted, and re-labelling is better than
thinning labels, because a voxel's representative point should carry a label
predicted from features at that scale rather than one inherited from a point
that is no longer there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.metrics import voxel_downsample  # noqa: E402
from twinworld.pointcloud import write_submission_ply  # noqa: E402


def read_points(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    if "classification" in {p.name for p in vertex.properties}:
        raise SystemExit(
            f"{path} carries a classification. Thinning a labelled cloud would keep "
            f"whichever label the surviving point happened to have; re-run "
            f"label_cloud.py --voxel instead")
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds", required=True, type=Path,
                        help="directory of <name>/point_cloud.ply exports")
    parser.add_argument("--pattern", default="tum_*",
                        help="which export directories to thin")
    parser.add_argument("--voxel", required=True, type=float,
                        help="target spacing in metres; 0.02 matches the metric and "
                             "is already what the exports carry")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sources = sorted(path for path in args.clouds.glob(f"{args.pattern}/point_cloud.ply"))
    if not sources:
        raise SystemExit(f"nothing matched {args.clouds / args.pattern}/point_cloud.ply")

    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"voxel": args.voxel, "source": str(args.clouds), "clouds": {}}
    print(f"{'export':28s} {'before':>12s} {'after':>12s} {'kept':>6s} "
          f"{'MB':>7s} {'secs':>6s}")
    print("-" * 78)

    for source in sources:
        name = source.parent.name
        started = time.time()
        points = read_points(source)
        thinned = voxel_downsample(points, args.voxel)
        destination = args.output / name / "point_cloud.ply"
        written = write_submission_ply(destination, thinned)
        megabytes = destination.stat().st_size / 1e6
        elapsed = time.time() - started
        receipt["clouds"][name] = {"before": int(len(points)), "after": written,
                                   "megabytes": round(megabytes, 1)}
        print(f"{name:28s} {len(points):>12,d} {written:>12,d} "
              f"{written / len(points):>5.1%} {megabytes:>7.1f} {elapsed:>6.0f}",
              flush=True)

    before = sum(row["before"] for row in receipt["clouds"].values())
    after = sum(row["after"] for row in receipt["clouds"].values())
    total = sum(row["megabytes"] for row in receipt["clouds"].values())
    print("-" * 78)
    print(f"{'total':28s} {before:>12,d} {after:>12,d} {after / before:>5.1%} "
          f"{total:>7.1f}")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
