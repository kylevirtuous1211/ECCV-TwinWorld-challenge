#!/usr/bin/env python3
"""Train one scene and write its test-pose renders, plus a held-out check.

With twelve training views there is no room for a validation split, so quality
is judged two ways instead: the train views themselves, which say whether the
fit converged, and a leave-one-out view held back from training, which says
whether it generalises. On the four dev scenes the test poses have reference
photos, so those are scored directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.rendering import psnr, score_pair, ssim  # noqa: E402
from twinworld.splat import (  # noqa: E402
    TrainConfig,
    load_checkpoint,
    load_frames,
    render_frames,
    save_checkpoint,
    train,
)


def read_init_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    if "red" in names:
        colours = np.stack([vertex["red"], vertex["green"], vertex["blue"]],
                           axis=1).astype(np.float64) / 255.0
    else:
        colours = np.full((len(points), 3), 0.5)
    return points, colours


def read_depth_prior(path):
    """Per-view monocular depth from dense_init.py --depth-maps, as float32."""
    import numpy as _np
    with _np.load(path) as archive:
        return {name: archive[name].astype(_np.float32) for name in archive.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("tum", "gold_coast"))
    parser.add_argument("--scene", required=True, help="e.g. scene_000")
    parser.add_argument("--model", default="3dgs", choices=("3dgs", "2dgs"))
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--holdout", type=int, default=-1,
                        help="index of a train view to hold back; -1 uses them all")
    parser.add_argument("--train-views", type=int, default=0,
                        help="keep only this many train views, spread out by farthest-point "
                             "sampling; 0 keeps all. The holdout is removed first, so a sweep "
                             "over this number is scored against one fixed unseen view")
    parser.add_argument("--normal-weight", type=float, default=None,
                        help="2DGS normal-consistency weight")
    parser.add_argument("--distortion-weight", type=float, default=None,
                        help="2DGS depth-distortion weight")
    parser.add_argument("--normal-target", default=None,
                        choices=("centre", "intersection"),
                        help="what the normal-consistency term agrees with. gsplat's "
                             "target is the finite-difference normal of a depth map that "
                             "is constant within each splat, so it points at the camera "
                             "and the term rotates splats to face it. 'intersection' "
                             "rebuilds it from the ray-disc hit")
    parser.add_argument("--normal-start", type=int, default=None)
    parser.add_argument("--distortion-start", type=int, default=None)
    parser.add_argument("--sh-degree", type=int, default=None,
                        help="spherical harmonic degree. The default 3 gives each gaussian "
                             "48 colour coefficients, fitted here from twelve photographs; "
                             "lowering it is the cheapest control on the 10 dB gap between "
                             "train and test PSNR")
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
                             "COLMAP points, which cover 0.07-0.20%% of the pixels; see "
                             "scripts/dense_init.py")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="render from these gaussians instead of training. The "
                             "renders that ship are the mean of three seeds, so "
                             "reproducing them from the released checkpoints needs a "
                             "path that loads rather than trains; this is it. Mirrors "
                             "export_geometry.py's flag of the same name")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="keep the trained gaussians as checkpoint.pt. Anything that "
                             "post-processes a render - pruning floaters, say - is an A/B "
                             "against the same gaussians, and rendering runs have a 3 dB "
                             "spread, so retraining to get the other arm measures the "
                             "spread instead of the change.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    directory = {"tum": "Data_TUM", "gold_coast": "Data_Goldcoast"}[args.dataset]
    scene_root = args.dataset_root / directory / args.scene
    config = TrainConfig(model=args.model, iterations=args.iterations)
    for name in ("normal_weight", "distortion_weight", "normal_start",
                 "distortion_start", "normal_target", "grow_grad2d", "prune_opacity",
                     "depth_prior_weight", "depth_prior_start", "seed",
                     "dropout", "opacity_noise", "sh_degree",
                     "strategy", "cap_max", "opacity_reg", "scale_reg"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)

    train_frames = load_frames(scene_root / "train", load_images=True)
    test_frames = load_frames(scene_root / "test", load_images=False)
    held_out = None
    if args.holdout >= 0:
        held_out = train_frames.pop(args.holdout % (len(train_frames) + 1))

    if args.train_views and args.train_views < len(train_frames):
        # Farthest-point sampling on camera centres. Taking the first N instead
        # would take a contiguous run of one flight line, which confounds "fewer
        # views" with "views from one direction" - the thing being measured.
        centres = np.array([np.linalg.inv(f.view_matrix.numpy())[:3, 3] for f in train_frames])
        chosen = [int(np.argmax(np.linalg.norm(centres - centres.mean(0), axis=1)))]
        while len(chosen) < args.train_views:
            distance = np.min(
                np.linalg.norm(centres[:, None] - centres[chosen][None], axis=2), axis=1)
            distance[chosen] = -1.0
            chosen.append(int(np.argmax(distance)))
        train_frames = [train_frames[i] for i in sorted(chosen)]

    init_path = args.init_cloud or scene_root / "train" / "sparse" / "0" / "points3D.ply"
    points, colours = read_init_points(init_path)

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train.log"
    lines: list[str] = []

    def log(message):
        print(message, flush=True)
        lines.append(str(message))

    started = time.time()
    if args.checkpoint:
        # Rendering only. `load_checkpoint` restores the model type and the SH
        # degree, which is everything `render_frames` reads; the training weights
        # in `config` are not used on this path and are left as parsed.
        log(f"{args.dataset}/{args.scene}  {args.model}  rendering from {args.checkpoint}")
        params, stored = load_checkpoint(args.checkpoint)
        config.model, config.sh_degree = stored.model, stored.sh_degree
        elapsed = time.time() - started
        log(f"loaded {len(params['means']):,} gaussians in {elapsed:.1f} s")
    else:
        log(f"{args.dataset}/{args.scene}  {args.model}  {args.iterations} iterations")
        priors = read_depth_prior(args.depth_prior) if args.depth_prior else None
        params = train(points, colours, train_frames, config, log=log, depth_priors=priors)
        elapsed = time.time() - started
        log(f"trained in {elapsed / 60:.1f} min, {len(params['means']):,} gaussians")

    receipt = {
        "dataset": args.dataset, "scene": args.scene, "model": args.model,
        "iterations": args.iterations, "train_views": len(train_frames),
        "gaussians": int(len(params["means"])), "minutes": round(elapsed / 60, 2),
        "init_points": int(len(points)), "init_cloud": str(init_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sh_degree": config.sh_degree,
        "depth_prior": str(args.depth_prior) if args.depth_prior else None,
        "depth_prior_weight": config.depth_prior_weight,
        "normal_weight": config.normal_weight, "distortion_weight": config.distortion_weight,
        "normal_start": config.normal_start, "distortion_start": config.distortion_start,
    }

    if args.save_checkpoint:
        save_checkpoint(params, config, args.output / "checkpoint.pt")
        log(f"saved {args.output / 'checkpoint.pt'}")

    renders = render_frames(params, test_frames, config)
    rgb_dir = args.output / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for stem, image in renders.items():
        Image.fromarray(image).save(rgb_dir / f"{stem}.png")
    log(f"wrote {len(renders)} test renders to {rgb_dir}")

    # Train views: did it fit at all?
    train_renders = render_frames(params, train_frames, config)
    train_scores = [psnr(train_renders[f.stem], (f.image.numpy() * 255).astype(np.uint8))
                    for f in train_frames]
    receipt["train_psnr"] = round(float(np.mean(train_scores)), 3)
    log(f"train psnr {receipt['train_psnr']:.2f}")

    # A view it never saw: does it generalise?
    if held_out is not None:
        rendered = render_frames(params, [held_out], config)[held_out.stem]
        truth = (held_out.image.numpy() * 255).astype(np.uint8)
        Image.fromarray(rendered).save(args.output / f"holdout_{held_out.stem}.png")
        Image.fromarray(truth).save(args.output / f"holdout_{held_out.stem}_reference.png")
        receipt["holdout_view"] = held_out.name
        receipt["holdout_psnr"] = round(psnr(rendered, truth), 3)
        receipt["holdout_ssim"] = round(ssim(rendered, truth), 4)
        log(f"holdout {held_out.name}: psnr {receipt['holdout_psnr']:.2f} "
            f"ssim {receipt['holdout_ssim']:.4f}")

    # Test poses carry reference photos on the dev scenes only.
    references = scene_root / "test" / "images"
    scored = []
    for frame in test_frames:
        for suffix in (".JPG", ".jpg", ".png"):
            path = references / f"{frame.stem}{suffix}"
            if path.exists():
                truth = np.asarray(Image.open(path).convert("RGB"))
                scored.append(score_pair(renders[frame.stem], truth))
                break
    if scored:
        receipt["test_psnr"] = round(float(np.mean([s.psnr for s in scored])), 3)
        receipt["test_ssim"] = round(float(np.mean([s.ssim for s in scored])), 4)
        lpips = [s.lpips for s in scored if s.lpips is not None]
        if len(lpips) == len(scored):
            receipt["test_lpips"] = round(float(np.mean(lpips)), 4)
        log(f"test psnr {receipt['test_psnr']:.2f} ssim {receipt['test_ssim']:.4f} "
            f"lpips {receipt.get('test_lpips', 'n/a')}")
    else:
        log("test poses have no reference photos - this scene is withheld")

    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    log_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
