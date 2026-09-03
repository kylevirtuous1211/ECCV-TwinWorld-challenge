#!/usr/bin/env python3
"""Build a dense initialisation cloud with COLMAP's MVS, from the released images.

The seed 2DGS starts from covers 0.07% to 0.20% of the pixels it has to explain,
and Sparse2DGS's ablation ([arXiv:2504.20378](https://arxiv.org/abs/2504.20378),
CVPR 2025) attributes 55% of its Chamfer gain to initialisation alone against 12%
for all four of its cross-view terms. What that paper initialises from is **MVS**,
and that is the part this repository has not tried.

The two cheaper routes to a denser seed are both measured and both lost:

- **A monocular pointmap.** MoGe-2 seeded `scene_000` with 5,239,319 points
  against COLMAP's 27,949 and geometry fell from F 0.5418 to 0.5172, rendering
  from 20.16 to 19.02 dB. `scripts/dense_init.py` records why: scored directly the
  seed reaches recall 0.304 at 5 cm where COLMAP reaches 0.009, and **precision
  0.163 where COLMAP reaches 0.459**. It is right about roughly where the surface
  is and wrong about exactly where, which a 5 cm threshold does not forgive.
- **Re-triangulating the released images.** `scripts/retriangulate.py` came back
  sparser on every TUM scene and further from the truth on every scene that ships
  it. The released model is a crop of a bundle adjustment over 819-plus images and
  whatever matched it had the discarded frames to chain through.

MVS is neither of those. It solves depth from **photometric consistency between
the real images** rather than predicting it, and it starts from the released
sparse model, which is the most accurate geometry in the dataset. So it can be
dense like the pointmap and precise like the crop, which is the combination
neither route reached.

    image_undistorter -> patch_match_stereo -> stereo_fusion

**The risk it does not clear, stated up front.** Patch match needs overlapping
views. TUM's released frames sit 1 to 8 apart in a continuous flight, which is
ample. Gold Coast's tracks link images a median of 62 frames apart and 310 at
most, across different passes of one flight, and that is the same gap that made
re-triangulation fail there. Expect TUM to work and Gold Coast to be the open
question, not the other way round.

The receipt carries the released cloud, the raw fusion and the seed that ships,
each measured against the ground truth two ways: distance (median, within 5 cm and
20 cm, the same question `retriangulate.py` asks) and `geometry_fscore` (which is
the number directly comparable to MoGe-2's F 0.3882 at precision 0.163).

    python scripts/dense_mvs.py --dataset-root <root> --dataset tum \\
        --scene scene_000 --output <dir> \\
        --colmap "micromamba run -p <env> colmap"

`--colmap` is a command, not just a path, because the conda-forge build needs its
own environment on `LD_LIBRARY_PATH` and `micromamba run` is how that is supplied.
`pycolmap`'s wheels ship no `patch_match_stereo`, so this shells out to the binary
rather than importing, and the binary must be a CUDA build - COLMAP has no CPU
patch match.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.dataset import load_scenes  # noqa: E402
from twinworld.metrics import crop_to_box, geometry_fscore, voxel_downsample  # noqa: E402

# COLMAP filters a fused point on how many views agree with it and at what angle.
# The defaults are tuned for hundreds of images; twelve is few enough that a
# permissive gate is the difference between a seed and an empty file, and a seed
# is allowed to be permissive because nothing downstream trusts it as a target.
DEFAULT_MIN_TRIANGULATION_ANGLE = 1.0
DEFAULT_FUSION_MIN_PIXELS = 3


def write_seed_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    """A seed cloud in the shape `read_init_points` expects: coordinates and colour.

    Identical to `dense_init.py`'s, and deliberately not `write_submission_ply`,
    which enforces the challenge's format and refuses colour. This file is never
    uploaded, it is training input.
    """
    vertex = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, axis].astype(np.float32)
    for channel, name in enumerate(("red", "green", "blue")):
        vertex[name] = colours[:, channel].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(str(path))


def read_coloured_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates and 0-255 colour from a PLY that carries both.

    COLMAP's fused cloud also carries normals, which nothing here uses: the seed
    contract is coordinates and colour, and 2DGS derives its own orientation.
    """
    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    if "red" not in names:
        return points, np.full((len(points), 3), 128, dtype=np.uint8)
    colours = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)
    return points, colours.astype(np.uint8)


def run_stage(command: list[str], name: str, log: Path) -> float:
    """One COLMAP stage, its output kept, its failure fatal and legible."""
    print(f"  {name} ...", flush=True)
    started = time.time()
    with log.open("a") as handle:
        handle.write(f"\n=== {name} ===\n{' '.join(shlex.quote(c) for c in command)}\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}; see {log}")
    print(f"  {name} took {elapsed:.1f}s", flush=True)
    return elapsed


def distance_to_truth(points: np.ndarray, truth: Path, crop: bool) -> dict | None:
    """How far each point sits from the ground truth, for the scenes that ship it.

    The same measurement `retriangulate.py` makes - ours to the truth, so adding
    points cannot flatter it - and by default cropped to the truth's own box,
    which that script does not do and which a dense cloud makes mandatory.

    MVS reconstructs everything the twelve cameras can see. The truth covers a
    tile inside that, so uncropped this measures how far the surrounding city is
    from the tile: on `tum/scene_000` the same fusion reads 74.8 cm uncropped and
    8.4 cm inside the box. `geometry_fscore` crops, so the uncropped number is
    also not the one any F here is computed from. Both are recorded because
    `retriangulate.py`'s rows are uncropped and the two experiments have to stay
    comparable.
    """
    if not truth.exists() or len(points) == 0:
        return None
    from scipy.spatial import cKDTree

    reference, _ = read_coloured_ply(truth)
    if crop:
        points = crop_to_box(points, reference)
        if len(points) == 0:
            return None
    distance, _ = cKDTree(reference).query(points, workers=-1)
    return {"points": int(len(points)),
            "median_cm": round(float(np.median(distance)) * 100, 2),
            "within_5cm": round(float((distance < 0.05).mean()), 4),
            "within_20cm": round(float((distance < 0.20).mean()), 4)}


def score_against_truth(points: np.ndarray, truth: Path) -> dict | None:
    """`geometry_fscore` of a seed, which is what decides whether it is worth training on.

    Distance alone cannot separate a seed that is dense and roughly right from one
    that is sparse and exactly right - the first has the recall and the second the
    precision, and only the F-score prices them against each other. MoGe-2's seed
    read F 0.3882 with recall 0.304 and precision 0.163 here; the released sparse
    points read precision 0.459 at recall 0.009.
    """
    if not truth.exists() or len(points) == 0:
        return None
    reference, _ = read_coloured_ply(truth)
    score = geometry_fscore(points, reference)
    per_threshold = {f"{int(threshold * 100)}cm": {
        "f": round(score.per_threshold[threshold], 4),
        "precision": round(score.precision[threshold], 4),
        "recall": round(score.recall[threshold], 4),
    } for threshold in sorted(score.per_threshold)}
    # `fscore` is the mean over the three thresholds, which is the number the
    # challenge reports and the one every F in HANDOVER.md is.
    return {"mean": round(score.fscore, 4), **per_threshold}


def measured(points: np.ndarray, truth: Path) -> dict:
    return {"points": int(len(points)),
            "against_truth": distance_to_truth(points, truth, crop=True),
            "against_truth_uncropped": distance_to_truth(points, truth, crop=False),
            "fscore": score_against_truth(points, truth)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("tum", "gold_coast"))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--colmap", default="colmap",
                        help="the COLMAP command, split as a shell word list. It must be "
                             "a CUDA build: patch match has no CPU implementation. The "
                             "conda-forge build needs its environment on the library path, "
                             'so this is usually "micromamba run -p <env> colmap"')
    parser.add_argument("--max-image-size", type=int, default=1600,
                        help="the released images are already 1600 px wide, so the default "
                             "asks for no downscale")
    parser.add_argument("--no-geom-consistency", action="store_true",
                        help="skip the second patch match pass, which re-filters each depth "
                             "map against its neighbours. Halves the time and costs the "
                             "precision the whole experiment is about")
    parser.add_argument("--min-triangulation-angle", type=float,
                        default=DEFAULT_MIN_TRIANGULATION_ANGLE,
                        help="degrees; COLMAP's own default of 1.0 assumes many views. A "
                             "drone at 40-210 m with frames seconds apart makes narrow "
                             "triangles, and this is the gate that discards them")
    parser.add_argument("--fusion-min-pixels", type=int, default=DEFAULT_FUSION_MIN_PIXELS,
                        help="how many views must agree before a point is fused. COLMAP's "
                             "default of 5 is a fifth of what twelve images can offer")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="thin the fused cloud before it becomes the seed, in metres. "
                             "0 keeps everything. MoGe-2's 5.2M-point seed grew 6.0M "
                             "gaussians and overfitted - train PSNR 28.5 to 36.6 - so seed "
                             "density is a knob and not a free win")
    parser.add_argument("--include-released", action="store_true",
                        help="add the released sparse points to the seed. They are the most "
                             "accurate geometry in the dataset and there are only tens of "
                             "thousands of them, so this cannot cost much and may anchor it")
    parser.add_argument("--keep-workspace", action="store_true",
                        help="keep the undistorted images and depth maps, which are several "
                             "GB a scene and are only needed to debug a bad fusion")
    args = parser.parse_args()

    scene = next((s for s in load_scenes(args.dataset_root)
                  if s.dataset == args.dataset and s.scene_id == args.scene), None)
    if scene is None:
        raise SystemExit(f"no {args.dataset}/{args.scene} under {args.dataset_root}")

    colmap = shlex.split(args.colmap)
    images = scene.root / "train" / "images"
    sparse = scene.root / "train" / "sparse" / "0"
    truth = scene.root / "3d_gt" / "point_cloud.ply"

    args.output.mkdir(parents=True, exist_ok=True)
    workspace = args.output / "dense"
    log = args.output / "colmap.log"
    log.write_text("")
    fused_path = args.output / "fused.ply"

    print(f"{args.dataset}/{args.scene} -> {args.output}", flush=True)

    seconds = {}
    seconds["undistort"] = run_stage(colmap + [
        "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(sparse),
        "--output_path", str(workspace),
        "--output_type", "COLMAP",
        "--max_image_size", str(args.max_image_size),
    ], "image_undistorter", log)

    seconds["patch_match"] = run_stage(colmap + [
        "patch_match_stereo",
        "--workspace_path", str(workspace),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "false" if args.no_geom_consistency else "true",
        "--PatchMatchStereo.filter_min_triangulation_angle",
        str(args.min_triangulation_angle),
    ], "patch_match_stereo", log)

    seconds["fusion"] = run_stage(colmap + [
        "stereo_fusion",
        "--workspace_path", str(workspace),
        "--workspace_format", "COLMAP",
        "--input_type", "photometric" if args.no_geom_consistency else "geometric",
        "--output_path", str(fused_path),
        "--StereoFusion.min_num_pixels", str(args.fusion_min_pixels),
    ], "stereo_fusion", log)

    fused_points, fused_colours = read_coloured_ply(fused_path)
    print(f"  fused {len(fused_points):,d} points", flush=True)

    seed_points, seed_colours = fused_points, fused_colours
    if args.voxel_size > 0:
        seed_points, index = voxel_downsample(seed_points, args.voxel_size, return_index=True)
        seed_colours = seed_colours[index]
        print(f"  thinned to {len(seed_points):,d} at {args.voxel_size} m", flush=True)

    released_points, released_colours = read_coloured_ply(sparse / "points3D.ply")
    if args.include_released:
        seed_points = np.concatenate([seed_points, released_points])
        seed_colours = np.concatenate([seed_colours, released_colours])
        print(f"  plus {len(released_points):,d} released -> {len(seed_points):,d}", flush=True)

    write_seed_ply(args.output / "points3D.ply", seed_points, seed_colours)

    receipt = {
        "dataset": args.dataset, "scene": args.scene,
        "max_image_size": args.max_image_size,
        "geom_consistency": not args.no_geom_consistency,
        "min_triangulation_angle": args.min_triangulation_angle,
        "fusion_min_pixels": args.fusion_min_pixels,
        "voxel_size": args.voxel_size,
        "include_released": args.include_released,
        "released": measured(released_points, truth),
        "fused": measured(fused_points, truth),
        "seed": measured(seed_points, truth),
        "seconds": {name: round(value, 1) for name, value in seconds.items()},
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))

    if not args.keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)

    for name in ("released", "fused", "seed"):
        row = receipt[name]
        distance, fscore = row["against_truth"], row["fscore"]
        line = f"  {name:9s} {row['points']:>10,d} points"
        if distance:
            line += (f"   in box {distance['points']:>9,d}"
                     f"   median {distance['median_cm']:>6.2f} cm"
                     f"   {distance['within_5cm']:.1%} within 5 cm")
        if fscore:
            at5 = fscore["5cm"]
            line += (f"   F {fscore['mean']:.4f}"
                     f"   F@5 {at5['f']:.4f}  P@5 {at5['precision']:.3f}"
                     f"  R@5 {at5['recall']:.3f}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
