#!/usr/bin/env python3
"""Export the surface cloud from a trained Gaussian scene, and score it.

The submission's geometry has never been produced by this repository. Training
wrote renders and a receipt and dropped the Gaussians on the floor, so the only
cloud that ever reached the scorer was the sparse COLMAP one the scene ships
with, at geometry_score 0.1603 over the four TUM dev scenes.

That score is recall-bound, not accuracy-bound. The sparse cloud occupies 12k to
33k of the ground truth's 4.7M to 11.9M two-centimetre voxels, so its precision
at 20 cm is already 0.69 to 0.91 while its recall at 5 cm is 0.002 to 0.019.
Rendering depth from the trained scene and unprojecting it is the step that
turns a scene into surface samples at the density the metric is asking for.

This script trains or loads a scene, renders depth from every pose it has, fuses
the result, writes the submission cloud, and - on the four dev scenes - scores it
against the ground truth immediately, so a change to the fusion is answered by a
number rather than by an argument.

Run it from the reconstruction environment:

    source scripts/env.sh
    python scripts/export_geometry.py --dataset-root <path> \
        --dataset tum --scene scene_000 --output <dir> --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.fusion import View, fuse  # noqa: E402
from twinworld.metrics import geometry_fscore  # noqa: E402
from twinworld.pointcloud import write_submission_ply  # noqa: E402

DATASET_DIRECTORY = {"tum": "Data_TUM", "gold_coast": "Data_Goldcoast"}

# Spacing for sampling a TSDF mesh. Finer than the metric's 2 cm voxel would be
# wasted, and the 5 cm recall threshold is what actually has to be met.
SURFACE_SPACING = 0.03


def read_init_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    if "red" in names:
        colours = np.stack([vertex["red"], vertex["green"], vertex["blue"]],
                           axis=1).astype(np.float64) / 255.0
    else:
        colours = np.full((len(points), 3), 0.5)
    return points, colours


def supersampled(frame, factor: int):
    """The same camera, rendered on a finer pixel grid.

    Depth is one sample per pixel, so the pixel grid is the sampling rate of the
    surface. At 1600 pixels across a 60 m scene that is 3.75 cm per pixel at
    nadir and worse at 45 degrees - coarser than the 2 cm voxel the metric
    counts in, so rays miss parts of the surface entirely. Scaling the
    intrinsics with the image keeps the same view and shrinks the gap between
    rays.
    """
    import dataclasses

    if factor == 1:
        return frame
    intrinsics = frame.intrinsics.clone()
    intrinsics[:2] *= factor            # fx, fy, cx, cy all scale with the grid
    return dataclasses.replace(frame, width=frame.width * factor,
                               height=frame.height * factor, intrinsics=intrinsics)


def as_view(frame) -> View:
    intrinsics = frame.intrinsics.numpy()
    return View(
        view_matrix=frame.view_matrix.numpy().astype(np.float64),
        fx=float(intrinsics[0, 0]), fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]), cy=float(intrinsics[1, 2]),
        width=frame.width, height=frame.height)


def scene_depth_limit(scene_root: Path, frames, margin: float = 1.25) -> float:
    """How far a depth reading may be before it cannot be this scene.

    A trained scene keeps stray Gaussians well behind everything real, and they
    are opaque enough to pass the alpha gate. On scene_000 - 45 by 72 by 24 m,
    cameras 80 m up - the rendered depth has a median of 44 m and a maximum of
    915 m, with 617k pixels beyond 200 m.

    Unprojection survives that by luck: the evaluator crops to the ground-truth
    box, so the strays are discarded before they are scored. A volume has no
    such crop and allocates blocks all the way out, which turned a few-minute
    integration into 43 GB and no result. It also means the submitted cloud
    carries hundreds of thousands of points that cannot score.

    The limit comes from the COLMAP points the scene ships with, which bound
    where the scene is, rather than from a constant that would be wrong for a
    scene of a different size.

    The COLMAP cloud has strays of its own, so no statistic of point distances
    is safe: on scene_000 the median point sits 45 m from the nearest camera,
    the 99th percentile at 414 m and the furthest at 1.4 km, which put a
    max-based limit at 2179 m - worse than the constant it replaced. So the
    bound comes from a robust box instead. Outliers cannot move a percentile
    box, and the question it answers is the right one: how far could a real
    surface be from a camera.
    """
    points, _ = read_init_points(scene_root / "train" / "sparse" / "0" / "points3D.ply")
    centres = np.array([np.linalg.inv(frame.view_matrix.numpy())[:3, 3] for frame in frames])

    lower = np.percentile(points, 2, axis=0)
    upper = np.percentile(points, 98, axis=0)
    corners = np.array(np.meshgrid(*zip(lower, upper))).reshape(3, -1).T
    furthest_corner = np.linalg.norm(corners[None, :, :] - centres[:, None, :], axis=2).max()
    return float(furthest_corner * margin)


def scene_box(scene_root: Path, margin: float = 10.0,
              percentile: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Where the scene is, as a box, from the COLMAP points.

    A radial depth limit is not enough on its own. Cutting scene_000 at 324 m
    still leaves hundreds of thousands of readings scattered through that
    sphere, and a volume allocates blocks for every one: 34 GB and no result
    after an hour. This is the same crop the evaluator applies at scoring time,
    moved to before the expensive part.

    The defaults are deliberately loose. Trimming to the 2nd percentile with a
    5 m margin cut 6.5% of scene_001's ground truth, which caps recall outright
    - a box meant to exclude strays at 900 m has no business shaving the scene.
    At the 0.5th percentile with 10 m all four dev scenes keep 100% of their
    ground truth, and the volume is still a twenty-fifth of an untrimmed box.
    """
    points, _ = read_init_points(scene_root / "train" / "sparse" / "0" / "points3D.ply")
    lower = np.percentile(points, percentile, axis=0) - margin
    upper = np.percentile(points, 100 - percentile, axis=0) + margin
    return lower, upper


def read_reference(scene_root: Path) -> np.ndarray | None:
    from plyfile import PlyData

    path = scene_root / "3d_gt" / "point_cloud.ply"
    if not path.exists():
        return None
    vertex = PlyData.read(str(path))["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def read_depth_prior(path):
    """Per-view monocular depth from dense_init.py --depth-maps, as float32."""
    import numpy as _np
    with _np.load(path) as archive:
        return {name: archive[name].astype(_np.float32) for name in archive.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_DIRECTORY))
    parser.add_argument("--scene", required=True, help="e.g. scene_000")
    parser.add_argument("--model", default="2dgs", choices=("3dgs", "2dgs"))
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="load these Gaussians instead of training")
    parser.add_argument("--views", default="train+test", choices=("train", "train+test"),
                        help="which poses to render depth from; test poses carry no "
                             "reference photo but their cameras are given")
    parser.add_argument("--depth", default="median", choices=("median", "expected"),
                        help="2DGS offers both; 3DGS only has expected")
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--min-views", type=int, default=3,
                        help="multi-view agreement required; 1 disables the check. "
                             "3 improves every dev scene; 6 scores higher on scene_000 "
                             "but loses on two of the four")
    parser.add_argument("--supersample", type=int, default=1,
                        help="render depth on an N times finer pixel grid")
    parser.add_argument("--fusion", default="points", choices=("points", "tsdf"),
                        help="unproject every view's depth, or fuse it in a volume")
    parser.add_argument("--tsdf-voxel", type=float, default=0.04)
    parser.add_argument("--depth-trunc", type=float, default=None,
                        help="metres beyond which a depth reading is discarded; "
                             "derived from the COLMAP extent when not given")
    parser.add_argument("--tsdf-trunc", type=float, default=4.0,
                        help="sdf_trunc as a multiple of the voxel; it must exceed "
                             "the disagreement between views or the volume keeps "
                             "both surfaces instead of averaging them")
    parser.add_argument("--distortion-weight", type=float, default=None,
                        help="2DGS depth-distortion weight. Off by default because it "
                             "costs rendering quality - but its job is geometric, and "
                             "no geometry metric existed when that was decided")
    parser.add_argument("--normal-weight", type=float, default=None)
    parser.add_argument("--normal-target", default=None,
                        choices=("centre", "intersection"),
                        help="what the normal-consistency term agrees with. gsplat's "
                             "target is the finite-difference normal of a depth map that "
                             "is constant within each splat, so it points at the camera "
                             "and the term rotates splats to face it. 'intersection' "
                             "rebuilds it from the ray-disc hit")
    parser.add_argument("--depth-readout", default="centre",
                        choices=("centre", "intersection"),
                        help="where along the ray to read depth. gsplat reports each "
                             "Gaussian's centre depth for every pixel it covers, which "
                             "is a known upstream bug (gsplat #863). 'intersection' "
                             "recovers the ray-disc hit and is worth +0.0829 over the "
                             "four dev scenes, measured on identical checkpoints")
    parser.add_argument("--sweep", action="store_true",
                        help="vary one setting at a time from the default and score each")
    parser.add_argument("--dropout", type=float, default=None,
                        help="share of gaussians dropped at random each training step. "
                             "Twelve photographs entangle them - train PSNR runs 10 to 13 dB "
                             "above test - and the cloud is fused from depth rendered at "
                             "poses that entanglement never had to explain. "
                             "arXiv:2508.12720")
    parser.add_argument("--opacity-noise", type=float, default=None,
                        help="multiplicative opacity noise, centred on one, same purpose")
    parser.add_argument("--seed", type=int, default=None,
                        help="the training seed. Two runs that differ only here agree "
                             "where the data determines the surface and disagree where "
                             "the fit was free to choose, which is a per-point confidence "
                             "no single run has")
    parser.add_argument("--depth-prior", type=Path, default=None,
                        help="npz of per-view monocular depth from dense_init.py "
                             "--depth-maps. Applied to the shape of the rendered depth "
                             "and never to its scale; see twinworld.splat.depth_prior_loss")
    parser.add_argument("--depth-prior-weight", type=float, default=None)
    parser.add_argument("--depth-prior-start", type=int, default=None)
    parser.add_argument("--strategy", default=None, choices=("default", "mcmc"),
                        help="how gaussians are added and removed. gsplat's default grows "
                             "on the view-space gradient, which is largest for whatever sits "
                             "nearest a camera and so manufactures near-camera floaters. "
                             "mcmc relocates dead gaussians onto live ones instead and never "
                             "reads that gradient; see twinworld/splat.py for what it is and "
                             "is not worth")
    parser.add_argument("--cap-max", type=int, default=None,
                        help="mcmc grows to this count rather than to a gradient threshold. "
                             "Set it to the default strategy's own final count to compare at "
                             "matched capacity, which is how the paper's tables are read")
    parser.add_argument("--opacity-reg", type=float, default=None)
    parser.add_argument("--scale-reg", type=float, default=None)
    parser.add_argument("--grow-grad2d", type=float, default=None,
                        help="gradient a gaussian needs before it splits. gsplat's 0.0002 "
                             "is tuned for hundreds of views; raising it caps capacity, "
                             "which is what the geometry is paying for")
    parser.add_argument("--prune-opacity", type=float, default=None,
                        help="opacity below which a gaussian is pruned, default 0.005")
    parser.add_argument("--init-cloud", type=Path, default=None,
                        help="seed the gaussians from this cloud instead of the scene's "
                             "COLMAP points; see scripts/dense_init.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from twinworld.splat import (
        TrainConfig, load_checkpoint, load_frames, render_depth_frames, save_checkpoint, train,
    )

    scene_root = args.dataset_root / DATASET_DIRECTORY[args.dataset] / args.scene
    args.output.mkdir(parents=True, exist_ok=True)

    train_frames = load_frames(scene_root / "train", load_images=True)
    test_frames = load_frames(scene_root / "test", load_images=False)

    if args.checkpoint:
        params, config = load_checkpoint(args.checkpoint)
        config.iterations = args.iterations
        print(f"loaded {len(params['means']):,} gaussians from {args.checkpoint}", flush=True)
    else:
        config = TrainConfig(model=args.model, iterations=args.iterations)
        for name in ("distortion_weight", "normal_weight", "normal_target",
                     "grow_grad2d", "prune_opacity",
                     "depth_prior_weight", "depth_prior_start", "seed",
                     "dropout", "opacity_noise",
                     "strategy", "cap_max", "opacity_reg", "scale_reg"):
            value = getattr(args, name)
            if value is not None:
                setattr(config, name, value)
        points, colours = read_init_points(
            args.init_cloud or scene_root / "train" / "sparse" / "0" / "points3D.ply")
        started = time.time()
        priors = read_depth_prior(args.depth_prior) if args.depth_prior else None
        params = train(points, colours, train_frames, config, depth_priors=priors)
        print(f"trained in {(time.time() - started) / 60:.1f} min, "
              f"{len(params['means']):,} gaussians", flush=True)
        save_checkpoint(params, config, args.output / "checkpoint.pt")

    frames = train_frames + (test_frames if args.views == "train+test" else [])
    if args.depth == "median" and config.model != "2dgs":
        raise SystemExit("median depth is a 2DGS output; use --depth expected for 3dgs")

    depth_trunc = args.depth_trunc or scene_depth_limit(scene_root, frames)
    scene_bounds = scene_box(scene_root)
    print(f"discarding depth beyond {depth_trunc:.0f} m, or outside "
          f"{np.round(scene_bounds[1] - scene_bounds[0], 1)} m", flush=True)

    renders: dict[int, tuple] = {}

    def render_at(factor: int):
        """Depth maps and matching cameras at one supersampling factor, once."""
        if factor not in renders:
            grid = [supersampled(frame, factor) for frame in frames]
            started = time.time()
            rendered = render_depth_frames(
                params, grid, config,
                intersection=args.depth_readout == "intersection")
            print(f"  rendered {len(grid)} views at {factor}x "
                  f"({grid[0].width}x{grid[0].height}) in {time.time() - started:.0f}s",
                  flush=True)
            renders[factor] = (
                [as_view(frame) for frame in grid],
                [rendered[frame.stem].alpha for frame in grid],
                {"median": [rendered[frame.stem].median_depth for frame in grid],
                 "expected": [rendered[frame.stem].depth for frame in grid]},
                [rendered[frame.stem].colour for frame in grid])
        return renders[factor]

    reference = read_reference(scene_root)
    receipt = {"dataset": args.dataset, "scene": args.scene, "model": config.model,
               "iterations": args.iterations, "views": len(frames),
               "gaussians": int(len(params["means"])),
               "distortion_weight": config.distortion_weight,
               "normal_weight": config.normal_weight}

    def evaluate(kind, alpha_min, min_views, supersample, label, fusion="points"):
        started = time.time()
        views, alphas, depths, images = render_at(supersample)
        colours = None
        if fusion == "tsdf":
            from twinworld.tsdf import fuse as volume_fuse
            cloud = volume_fuse(depths[kind], alphas, views, alpha_min=alpha_min,
                                voxel_length=args.tsdf_voxel,
                                trunc_factor=args.tsdf_trunc, spacing=SURFACE_SPACING,
                                depth_trunc=depth_trunc, bounds=scene_bounds)
        else:
            # The colour rides along at no cost: the rasteriser rendered it in the
            # same call as the depth, and every step of the fusion is
            # index-preserving. TSDF resamples onto a volume, so a per-point
            # correspondence does not survive it and none is asked for.
            cloud, colours = fuse(depths[kind], alphas, views, alpha_min=alpha_min,
                                  voxel_size=args.voxel, min_views=min_views,
                                  far=depth_trunc, images=images)
        row = {"fusion": fusion, "depth": kind, "alpha_min": alpha_min,
               "min_views": min_views, "supersample": supersample,
               "points": int(len(cloud))}
        if reference is not None and len(cloud):
            score = geometry_fscore(cloud, reference)
            row.update(fscore=round(score.fscore, 4),
                       precision_5=round(score.precision[0.05], 4),
                       recall_5=round(score.recall[0.05], 4),
                       precision_10=round(score.precision[0.10], 4),
                       recall_10=round(score.recall[0.10], 4),
                       precision_20=round(score.precision[0.20], 4),
                       recall_20=round(score.recall[0.20], 4),
                       submitted=int(score.predicted_points))
        # Timed after scoring, not before: the fusion is seconds and the scoring
        # is minutes, so timing only the fusion reports a misleading number.
        elapsed = time.time() - started
        row["seconds"] = round(elapsed, 1)
        print(f"  {label:28s} {row['points']:11,d} "
              f"{row.get('precision_5', float('nan')):6.3f} "
              f"{row.get('recall_5', float('nan')):6.3f} "
              f"{row.get('precision_20', float('nan')):6.3f} "
              f"{row.get('recall_20', float('nan')):6.3f} "
              f"{row.get('fscore', float('nan')):7.4f} {elapsed:6.0f}", flush=True)
        return cloud, colours, row

    header = (f"  {'setting':28s} {'points':>11s} {'P@5':>6s} {'R@5':>6s} "
              f"{'P@20':>6s} {'R@20':>6s} {'F':>7s} {'secs':>6s}")
    print(f"\n{args.dataset}/{args.scene}  {config.model}  {len(frames)} views")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    if args.sweep:
        # One factor at a time from the default, not a full grid. A grid of this
        # costs hours because each cell builds a KD-tree over 25M points, and it
        # answers a question nobody asked - what is wanted is how much each knob
        # is worth on its own.
        base = dict(kind=args.depth, alpha_min=args.alpha_min,
                    min_views=args.min_views, supersample=args.supersample,
                    fusion=args.fusion)
        variations = [("default", {})]
        variations += [(f"alpha {a}", {"alpha_min": a})
                       for a in (0.1, 0.3, 0.7, 0.9) if a != base["alpha_min"]]
        # Agreement is the only per-point confidence the fusion has, and it is
        # the whole of what stands between a 17.5M-point cloud and a 4.7M-voxel
        # truth. `--min-views 3` was picked over 6 on the centre-depth clouds,
        # where a splat's points landed in a plane facing the camera and so
        # agreed with each other for the wrong reason. It is worth re-asking now
        # that the points follow the disc they came from.
        variations += [(f"agree >= {v}", {"min_views": v})
                       for v in (2, 3, 4, 5, 6) if v != base["min_views"]]
        variations += [(f"supersample {s}x", {"supersample": s}) for s in (2,)]
        if config.model == "2dgs":
            other = "expected" if base["kind"] == "median" else "median"
            variations.append((f"{other} depth", {"kind": other}))
        # TSDF attacks the same thickness that agreement filtering does, and the
        # tilt bias it carries is smallest on a fine pixel grid, so it is tried
        # both ways round.
        variations += [("tsdf", {"fusion": "tsdf"}),
                       ("tsdf, supersample 2x", {"fusion": "tsdf", "supersample": 2})]

        best_cloud, best_colours, best_row = None, None, None
        for label, override in variations:
            settings = {**base, **override}
            cloud, colours, row = evaluate(
                settings["kind"], settings["alpha_min"], settings["min_views"],
                settings["supersample"], label, fusion=settings["fusion"])
            row["label"] = label
            rows.append(row)
            if best_row is None or row.get("fscore", -1) > best_row.get("fscore", -1):
                best_cloud, best_colours, best_row = cloud, colours, row
        cloud, colours, chosen = best_cloud, best_colours, best_row
        print(f"\n  best: {chosen.get('label')}  F={chosen.get('fscore')}")
    else:
        cloud, colours, chosen = evaluate(
            args.depth, args.alpha_min, args.min_views, args.supersample,
            f"{args.fusion}, {args.depth}, a>={args.alpha_min}, "
            f"v>={args.min_views}, {args.supersample}x", fusion=args.fusion)
        rows.append(chosen)

    written = write_submission_ply(args.output / "point_cloud.ply", cloud)
    if colours is not None and len(colours) == len(cloud):
        # A sidecar, not a PLY field. The submission format is x, y, z plus a
        # classification on Gold Coast and nothing else, and write_submission_ply
        # refuses anything the evaluator ignores, so the colour has to live
        # beside the cloud rather than in it.
        np.save(args.output / "colour.npy", colours.astype(np.float32))
        receipt["colour"] = "colour.npy"
    receipt["fusion"] = chosen
    receipt["sweep"] = rows
    receipt["written_points"] = written
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"\nwrote {written:,} points to {args.output / 'point_cloud.ply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
