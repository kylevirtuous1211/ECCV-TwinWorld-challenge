#!/usr/bin/env python3
"""Score a submission on all three benchmarks, offline, as often as we like.

Six scenes ship with what is needed to grade ourselves: TUM 000 to 003 carry
reference photos and a ground-truth cloud, Gold Coast 009 and 010 carry
reference photos and a labelled cloud. The other seven are withheld, so they
appear in a submission but can never be scored here.

    rendering_score = mean(PSNR / 50 capped at 1,  SSIM,  1 - LPIPS)   all scenes
    geometry_score  = mean(F-score @ 5, 10, 20 cm)                     TUM only
    semantic_score  = mIoU 3D over the five classes                    Gold Coast only
    Final Score     = the three added, so 3.0 is the maximum

LPIPS needs torch, so run this from the twinworld environment (`source
scripts/env.sh`) for a complete rendering score. Without it the two other terms
are still reported and the rendering score is withheld rather than quietly
averaged over two terms.

These follow the stated definitions rather than the organisers' code, so they
are a yardstick for comparing our own attempts, not a prediction of the
leaderboard's absolute numbers.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.dataset import SEMANTIC_CLASSES, load_scenes  # noqa: E402
from twinworld.metrics import (  # noqa: E402
    SEMANTIC_CLASS_INDICES,
    confusion,
    geometry_fscore,
    pooled_miou,
    transfer_labels,
)
from twinworld.rendering import combine, score_pair  # noqa: E402


def read_cloud(source) -> tuple[np.ndarray, np.ndarray | None]:
    vertex = PlyData.read(source)["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    labels = np.asarray(vertex["classification"]) if "classification" in names else None
    return points, labels


class Source:
    """A submission zip, or the sparse COLMAP baseline when none is given."""

    def __init__(self, zip_path: Path | None):
        self.archive = zipfile.ZipFile(zip_path) if zip_path else None
        self.label = zip_path.name if zip_path else "sparse COLMAP baseline (no renders)"

    def cloud(self, scene):
        if self.archive is None:
            return read_cloud(str(scene.root / "train" / "sparse" / "0" / "points3D.ply"))
        entry = f"{scene.dataset}/{scene.scene_id}/3D_point_cloud/point_cloud.ply"
        if entry not in self.archive.namelist():
            return None, None
        return read_cloud(BytesIO(self.archive.read(entry)))

    def render(self, scene, stem):
        if self.archive is None:
            return None
        entry = f"{scene.dataset}/{scene.scene_id}/rgb/{stem}.png"
        if entry not in self.archive.namelist():
            return None
        return np.asarray(Image.open(BytesIO(self.archive.read(entry))).convert("RGB"))


def reference_photo(scene, stem) -> np.ndarray | None:
    directory = scene.root / "test" / "images"
    for candidate in (f"{stem}.JPG", f"{stem}.jpg", f"{stem}.png"):
        path = directory / candidate
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--no-lpips", action="store_true",
                        help="skip LPIPS; the rendering score is then withheld")
    args = parser.parse_args()

    scenes = load_scenes(args.dataset_root)
    source = Source(args.submission)
    print(f"scoring {source.label}\n")

    rendering, geometry, confusions = [], [], []
    print(f"{'scene':22s} {'PSNR':>7s} {'SSIM':>6s} {'LPIPS':>6s} {'F-score':>8s} {'note'}")
    print("-" * 74)

    for scene in scenes:
        if not (scene.root / "3d_gt" / "point_cloud.ply").exists():
            continue
        row = {"psnr": "-", "ssim": "-", "lpips": "-", "f": "-", "note": ""}

        frames = []
        for frame in scene.frames:
            truth = reference_photo(scene, frame.stem)
            rendered = source.render(scene, frame.stem)
            if truth is None or rendered is None:
                continue
            if rendered.shape != truth.shape:
                row["note"] = "render/reference shape mismatch"
                continue
            frames.append(score_pair(rendered, truth, with_lpips=not args.no_lpips))
        if frames:
            merged = combine(frames)
            rendering.append(merged)
            row["psnr"] = f"{merged.psnr:.2f}"
            row["ssim"] = f"{merged.ssim:.4f}"
            row["lpips"] = "-" if merged.lpips is None else f"{merged.lpips:.4f}"
        elif source.archive is not None:
            row["note"] = "no renders found"

        predicted, labels = source.cloud(scene)
        reference, reference_labels = read_cloud(str(scene.root / "3d_gt" / "point_cloud.ply"))
        if predicted is None:
            row["note"] = "point cloud missing from the submission"
        elif scene.dataset == "tum":
            score = geometry_fscore(predicted, reference)
            geometry.append(score.fscore)
            row["f"] = f"{score.fscore:.4f}"
        else:
            if labels is None:
                labels = np.full(len(predicted), 255, np.uint8)
                row["note"] = (row["note"] + " unlabelled cloud").strip()
            confusions.append(confusion(reference_labels,
                                        transfer_labels(predicted, labels, reference)))

        print(f"{scene.dataset + '/' + scene.scene_id:22s} {row['psnr']:>7s} {row['ssim']:>6s} "
              f"{row['lpips']:>6s} {row['f']:>8s} {row['note']}")

    print()
    parts = {}
    if rendering:
        overall = combine(rendering)
        parts["rendering_score"] = overall.score
        detail = (f"PSNR {overall.psnr:.2f}, SSIM {overall.ssim:.4f}, "
                  f"LPIPS {'n/a' if overall.lpips is None else f'{overall.lpips:.4f}'}")
        shown = "withheld (no LPIPS)" if overall.score is None else f"{overall.score:.4f}"
        print(f"rendering_score  {shown:>19s}   {detail}")
    if geometry:
        parts["geometry_score"] = float(np.mean(geometry))
        print(f"geometry_score   {parts['geometry_score']:19.4f}   "
              f"mean over {len(geometry)} TUM scenes")
    if confusions:
        miou, per_class = pooled_miou(confusions)
        parts["semantic_score"] = miou
        per = ", ".join(
            f"{SEMANTIC_CLASSES[i]} {'absent' if np.isnan(per_class[i]) else f'{per_class[i]:.3f}'}"
            for i in SEMANTIC_CLASS_INDICES)
        print(f"semantic_score   {miou:19.4f}   {per}")

    known = [v for v in parts.values() if v is not None]
    total = sum(known)
    missing = [k for k, v in parts.items() if v is None] + \
              [k for k in ("rendering_score", "geometry_score", "semantic_score") if k not in parts]
    print(f"\nFinal Score      {total:19.4f}   out of 3.0"
          + (f"  (missing: {', '.join(missing)})" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
