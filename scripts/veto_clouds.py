#!/usr/bin/env python3
"""Filter a directory of clouds by what a second reconstruction will vouch for.

Measured on `scene_000` against the shipped 2DGS cloud, with Depth Anything 3 as
the second opinion:

    method       points        F     P@5    R@5   P@20   R@20
    ours      8,571,355   0.5513   0.349  0.442  0.662  0.713
    keep      4,682,028   0.6037   0.520  0.376  0.836  0.670
    fill+keep 5,961,378   0.6056   0.426  0.428  0.770  0.742

Dropping every point the other reconstruction has nothing near takes precision
at 5 cm from 0.349 to 0.520 and F from 0.5513 to 0.6056, and it does so while
*shrinking* the cloud by a third. Both halves of that matter here: the scorer
refused 205.8M points and accepted 111.8M, so until now density and accuracy
were in direct conflict. A veto is the first thing measured that improves both.

Why a second reconstruction can veto at all, when photo-consistency cannot:
this asks whether two methods that fail differently agree, not whether a patch
matches across views. 2DGS is fitted to these twelve photographs and DA3 has
learned what buildings look like from elsewhere, so a point they both place is
supported by both the data and the prior, and a point only one places is
supported by whichever of the two happened to be free to invent it.

    python scripts/veto_clouds.py --clouds <geom_isect> --other <da3> \\
        --pattern 'tum_*_isect' --output <dir>

The output mirrors the input's directory layout, so `make_submission.py
--clouds` consumes it unchanged, exactly like `downsample_clouds.py`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.metrics import voxel_downsample  # noqa: E402


def read_points(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def write_points(path: Path, points: np.ndarray) -> None:
    vertex = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, axis].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False,
            byte_order="<").write(str(path))


def scene_of(name: str) -> str:
    """`tum_scene_004_isect` and `gc_scene_009` both answer scene_004 / scene_009."""
    match = re.search(r"scene_\d+", name)
    if not match:
        raise SystemExit(f"cannot tell which scene {name} is")
    return match.group(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds", required=True, type=Path,
                        help="directory of <name>/point_cloud.ply to filter")
    parser.add_argument("--other", required=True, type=Path, nargs="+",
                        help="one or more directories of second opinions, matched by scene "
                             "id. With several, --min-agree decides how many must vouch")
    parser.add_argument("--min-agree", type=int, default=1,
                        help="how many of the --other reconstructions must have a point "
                             "within --keep-radius. One second opinion can be confidently "
                             "wrong - Depth Anything 3 scores 0.5172 on scene_000 and "
                             "0.1731 on scene_002 - so a vote is the honest form of this")
    parser.add_argument("--pattern", default="tum_*",
                        help="which subdirectories of --clouds to take")
    parser.add_argument("--keep-radius", type=float, default=0.10,
                        help="drop our points with nothing this close in the other cloud")
    parser.add_argument("--fill-radius", type=float, default=0.0,
                        help="add the other's points this far from any of ours. 0 disables "
                             "filling, which is the safer default: it wins by 0.002 on "
                             "scene_000 and costs 1.3M points")
    parser.add_argument("--voxel", type=float, default=0.02,
                        help="both sides are thinned to this before the comparison")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sources = sorted(path for path in args.clouds.iterdir()
                     if path.is_dir() and fnmatch.fnmatch(path.name, args.pattern))
    if not sources:
        raise SystemExit(f"no subdirectory of {args.clouds} matches {args.pattern!r}")

    others = []
    for directory in args.other:
        others.append({scene_of(path.name): path for path in directory.iterdir()
                       if path.is_dir() and (path / "point_cloud.ply").exists()})
    if args.min_agree > len(others):
        raise SystemExit(
            f"--min-agree {args.min_agree} needs at least that many --other directories, "
            f"and {len(others)} were given")

    print(f"{'cloud':28s} {'before':>12s} {'kept':>12s} {'added':>10s} {'after':>12s}  "
          f"{'%':>5s} {'MB':>7s} {'secs':>5s}")
    receipt = {"keep_radius": args.keep_radius, "fill_radius": args.fill_radius,
               "voxel": args.voxel, "source": str(args.clouds),
               "other": [str(path) for path in args.other],
               "min_agree": args.min_agree, "clouds": {}}
    for source in sources:
        scene = scene_of(source.name)
        missing_from = [index for index, lookup in enumerate(others) if scene not in lookup]
        if missing_from:
            raise SystemExit(
                f"{source.name} is {scene} and {args.other[missing_from[0]]} has no cloud "
                f"for it. Every scene in a submission needs one, so this refuses rather "
                f"than quietly shipping an unfiltered cloud beside filtered ones.")
        started = time.time()
        ours = read_points(source / "point_cloud.ply")

        votes = np.zeros(len(ours), dtype=np.int16)
        added = np.empty((0, 3))
        for lookup in others:
            theirs = voxel_downsample(read_points(lookup[scene] / "point_cloud.ply"),
                                      args.voxel)
            distance, _ = cKDTree(theirs).query(
                ours, k=1, distance_upper_bound=args.keep_radius, workers=-1)
            votes += np.isfinite(distance)
            if args.fill_radius > 0:
                our_tree = cKDTree(voxel_downsample(ours, args.voxel))
                back, _ = our_tree.query(theirs, k=1,
                                         distance_upper_bound=args.fill_radius, workers=-1)
                added = np.concatenate([added, theirs[~np.isfinite(back)]])
        kept = ours[votes >= args.min_agree]

        result = voxel_downsample(np.concatenate([kept, added]), args.voxel)
        destination = args.output / source.name / "point_cloud.ply"
        write_points(destination, result)
        megabytes = destination.stat().st_size / 1e6
        seconds = time.time() - started
        receipt["clouds"][source.name] = {
            "before": int(len(ours)), "kept": int(len(kept)), "added": int(len(added)),
            "after": int(len(result)), "megabytes": round(megabytes, 1)}
        print(f"{source.name:28s} {len(ours):>12,} {len(kept):>12,} {len(added):>10,} "
              f"{len(result):>12,}  {100 * len(result) / max(len(ours), 1):5.1f} "
              f"{megabytes:7.1f} {seconds:5.0f}", flush=True)

    total_before = sum(row["before"] for row in receipt["clouds"].values())
    total_after = sum(row["after"] for row in receipt["clouds"].values())
    print(f"{'total':28s} {total_before:>12,} {'':>12s} {'':>10s} {total_after:>12,}  "
          f"{100 * total_after / max(total_before, 1):5.1f}")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
