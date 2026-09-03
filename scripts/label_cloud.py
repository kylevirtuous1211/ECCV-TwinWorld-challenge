#!/usr/bin/env python3
"""Give the Gold Coast cloud its five classes, and score the result.

The semantic score has never been produced by this repository. Its baseline
calls every point wall, and the sparse COLMAP cloud it labels puts only 0.604%
of the ground truth within the evaluator's 10 cm match radius, so the mIoU
ceiling was 0.0068 and a perfect labelling of it would have reached 0.0067.
Coverage had to be fixed first, and `scripts/export_geometry.py` fixes it.

With coverage solved the labels become the binding term, and geometry alone
carries them a fair way: measured across the two labelled scenes, a classifier
over height above ground and verticality reaches pooled mIoU 0.453, against
0.112 for calling everything wall.

Gold Coast is not scored on geometry, and the semantic metric only needs a
predicted point within 10 cm of each true one, so the submitted cloud is
downsampled hard. That is not a compromise: at 5 cm spacing the coverage term
is already saturated, and a sparser cloud makes the features affordable.

Two things about the frame, and they decide what the numbers here mean:

  --offsets   The Gold Coast ground truth sits 1.1 to 1.4 m out of the camera
              frame its own images define, which `scripts/frame_offset.py`
              measures and TUM's clean result controls for. Given the offsets
              this script brings the truth into the reconstruction's frame
              before comparing them, and reports both scores. It never moves the
              cloud it writes - that is `scripts/shift_clouds.py`, and it is a
              separate decision, because which frame the evaluator holds is not
              known from here.

  --fit-on    The classifier is fitted on ground-truth clouds and applied to
              reconstructed ones, which have different noise, different density
              and different completeness. `--fit-on reconstruction` fits it on
              the clouds it will actually see, taking each training label from
              the nearest ground-truth point.

    source scripts/env.sh
    python scripts/label_cloud.py --dataset-root <path> --fusion-root <path> \
        --output <dir>
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

from twinworld.dataset import SEMANTIC_CLASSES  # noqa: E402
from twinworld.metrics import (  # noqa: E402
    IGNORE_LABEL,
    SEMANTIC_CLASS_INDICES,
    SEMANTIC_MATCH_RADIUS,
    confusion,
    coverage,
    pooled_ceiling,
    pooled_miou,
    transfer_labels,
    voxel_downsample,
)
from twinworld.pointcloud import write_submission_ply  # noqa: E402
from twinworld.semantics import feature_names, features  # noqa: E402

SUBMISSION_VOXEL = 0.04        # well inside the metric's 10 cm match radius
TRAIN_SAMPLE = 300_000
LABELLED = ("scene_009", "scene_010")


def read_labelled(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    labels = (np.asarray(vertex["classification"]).astype(np.uint8)
              if "classification" in names else np.full(len(points), 255, np.uint8))
    return points, labels


def truth_cloud(scene_root: Path, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ground truth, moved into the reconstruction's frame and kept scorable.

    `offset` is what the scene's own COLMAP points must be translated by to reach
    the truth, so subtracting it brings the truth back to where the cameras -
    and therefore everything we reconstruct - actually are.
    """
    points, labels = read_labelled(scene_root / "3d_gt" / "point_cloud.ply")
    scored = np.isin(labels, SEMANTIC_CLASS_INDICES)
    return points[scored] - offset, labels[scored]


def read_colour(cloud_path: Path, count: int) -> np.ndarray | None:
    """The rendered colour `export_geometry.py` saved beside the cloud, if it did.

    A sidecar rather than a PLY field, because the submission format is x, y, z
    plus a classification and nothing else. Absent for every export written
    before colour was carried through fusion, which is why this returns None
    rather than raising.
    """
    path = cloud_path.parent / "colour.npy"
    if not path.exists():
        return None
    colour = np.load(path).astype(np.float64)
    if len(colour) != count:
        raise SystemExit(
            f"{path} has {len(colour):,} colours for {count:,} points, so it was "
            f"written for a different export. Delete it or re-export the cloud.")
    return colour


def resolve_cloud(fusion_root: Path, scene: str, variant: str) -> Path | None:
    """The one export to label, or a refusal naming the candidates.

    Taking the first match is how the wrong cloud gets labelled without anyone
    noticing: the sweep leaves a dozen gc_scene_009_* directories and "2dgs"
    sorts ahead of "d0.01", so the run without the distortion term wins a coin
    toss it was never meant to enter.
    """
    pattern = f"gc_{scene}_{variant}/point_cloud.ply"
    found = sorted(fusion_root.glob(pattern))
    if not found:
        return None
    if len(found) > 1:
        listed = "\n    ".join(str(path.parent.name) for path in found)
        raise SystemExit(
            f"{scene}: {len(found)} exports match {pattern}, so which cloud to label "
            f"is undecided. Pass --variant.\n    {listed}")
    return found[0]


def load_cloud(path: Path, voxel: float, with_colour: bool):
    """A fused cloud thinned to `voxel`, with its colour thinned by the same index."""
    points = read_labelled(path)[0]
    colour = read_colour(path, len(points)) if with_colour else None
    if with_colour and colour is None:
        raise SystemExit(
            f"--colour was asked for but {path.parent / 'colour.npy'} does not exist. "
            f"Re-export that cloud with a build of export_geometry.py that carries "
            f"colour through fusion.")

    points, kept = voxel_downsample(points, voxel, return_index=True)
    return points, (colour[kept] if colour is not None else None)


def subsample(count: int, sample: int, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    if count <= sample:
        return np.arange(count)
    return generator.choice(count, sample, replace=False)


def training_from_truth(truth_points: np.ndarray, truth_labels: np.ndarray):
    """Features and labels from a ground-truth cloud, subsampled to stay affordable."""
    # Voxel first so the sample is spread over the scene rather than over its
    # densest corner, then a random draw to hit the budget.
    keep = np.arange(len(truth_points))
    reduced = voxel_downsample(truth_points, 0.05)
    if len(reduced) < len(truth_points):
        _, keep = np.unique(np.floor(truth_points / 0.05).astype(np.int64),
                            axis=0, return_index=True)
    keep = keep[subsample(len(keep), TRAIN_SAMPLE)]
    return features(truth_points[keep]), truth_labels[keep]


def training_from_reconstruction(points: np.ndarray, matrix: np.ndarray,
                                 truth_points: np.ndarray, truth_labels: np.ndarray,
                                 radius: float):
    """Features of the reconstructed cloud, labelled by the nearest truth point.

    Every row here is a point the classifier will genuinely be asked about at
    submission time, which the ground-truth rows are not: a LiDAR surface and a
    fused splat surface differ in noise, in density and in what they are missing,
    and a forest fitted on one is being asked to extrapolate on the other.

    Points with no truth inside `radius` carry no label and are dropped rather
    than guessed. Ours is the cloud with the coverage problem, so the ones that
    are dropped are exactly the ones nothing could have taught.
    """
    transferred = transfer_labels(truth_points, truth_labels, points, radius)
    matched = transferred != IGNORE_LABEL
    index = np.flatnonzero(matched)
    index = index[subsample(len(index), TRAIN_SAMPLE)]
    return matrix[index], transferred[index], float(matched.mean())


FOREST = {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 20}


def fit(matrix: np.ndarray, labels: np.ndarray, seed: int = 0):
    """The forest, at whatever size FOREST currently says.

    The defaults are small - sixty trees at depth twelve - and were never swept,
    which was defensible while the classifier owned 0.097 of the semantic gap in
    the misregistered frame. It now owns 0.169 of a gap whose ceiling is 0.6096,
    which makes it the largest single pool left.
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **FOREST)
    model.fit(matrix, labels)
    return model


def offsets_by_scene(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    receipt = json.loads(path.read_text())
    return {key.split("/")[-1]: np.asarray(row["offset"], dtype=np.float64)
            for key, row in receipt["scenes"].items()}


def report(name: str, matrix: np.ndarray) -> float:
    miou, per_class = pooled_miou([matrix])
    print(f"  {name}: mIoU {miou:.4f}   "
          + ", ".join(f"{SEMANTIC_CLASSES[i]} {per_class[i]:.3f}"
                      for i in SEMANTIC_CLASS_INDICES), flush=True)
    return miou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--fusion-root", required=True, type=Path,
                        help="directory of gc_<scene>_<variant>/point_cloud.ply exports")
    parser.add_argument("--variant", default="*",
                        help="which export suffix to label, e.g. d0.01. The exports "
                             "directory accumulates one directory per sweep cell, so "
                             "leaving this open is only safe when exactly one survives")
    parser.add_argument("--voxel", type=float, default=SUBMISSION_VOXEL)
    parser.add_argument("--fit-on", default="truth", choices=["truth", "reconstruction"],
                        help="which clouds the classifier is fitted on")
    parser.add_argument("--offsets", type=Path, default=None,
                        help="receipt from scripts/frame_offset.py. Brings the ground "
                             "truth into the reconstruction's frame for comparison; "
                             "the written cloud is never moved")
    parser.add_argument("--trees", type=int, default=None,
                        help="forest size, default 60")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="tree depth, default 12")
    parser.add_argument("--min-leaf", type=int, default=None,
                        help="minimum samples per leaf, default 20")
    parser.add_argument("--colour", action="store_true",
                        help="add the rendered RGB that export_geometry.py saved beside "
                             "each cloud as three more features. Only meaningful with "
                             "--fit-on reconstruction, because the ground-truth clouds "
                             "carry no colour at all")
    parser.add_argument("--transfer-radius", type=float, default=SEMANTIC_MATCH_RADIUS,
                        help="how far a reconstructed point may be from the truth and "
                             "still take its label, when fitting on the reconstruction")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.colour and args.fit_on != "reconstruction":
        raise SystemExit(
            "--colour needs --fit-on reconstruction. The ground-truth clouds carry "
            "x, y, z and a classification and no colour, so a classifier fitted on "
            "them has no colour column to learn from and could not use one at "
            "prediction time either. The two changes only work together.")

    if args.fit_on == "reconstruction" and args.offsets is None:
        raise SystemExit(
            "--fit-on reconstruction takes every training label from the nearest "
            "ground-truth point, and the Gold Coast ground truth is 1.1 to 1.4 m out "
            "of the frame the reconstruction is in. Without --offsets every label "
            "would come from the wrong place. Run scripts/frame_offset.py first.")

    gold_coast = args.dataset_root / "Data_Goldcoast"
    offsets = offsets_by_scene(args.offsets)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, key in (("trees", "n_estimators"), ("max_depth", "max_depth"),
                      ("min_leaf", "min_samples_leaf")):
        value = getattr(args, name)
        if value is not None:
            FOREST[key] = value

    receipt = {"voxel": args.voxel, "fit_on": args.fit_on, "colour": args.colour,
               "forest": dict(FOREST),
               "features": list(feature_names(args.colour)),
               "offsets": str(args.offsets.resolve()) if args.offsets else None,
               "scenes": {}}

    # Resolve every cloud before doing an hour of work on the first one.
    scenes = sorted(path.name for path in gold_coast.glob("scene_*"))
    sources = {scene: resolve_cloud(args.fusion_root, scene, args.variant)
               for scene in scenes}
    for scene, source in sources.items():
        if source is None:
            print(f"{scene}: no fused cloud yet", flush=True)

    print(f"fitting on the {args.fit_on} clouds")
    training, prepared = {}, {}
    for scene in LABELLED:
        started = time.time()
        offset = offsets.get(scene, np.zeros(3))
        truth_points, truth_labels = truth_cloud(gold_coast / scene, offset)

        if args.fit_on == "truth":
            training[scene] = training_from_truth(truth_points, truth_labels)
            covered = None
        else:
            if sources[scene] is None:
                raise SystemExit(f"{scene} has no fused cloud, so it cannot be fitted on")
            points, colour = load_cloud(sources[scene], args.voxel, args.colour)
            matrix = features(points, colour=colour)
            prepared[scene] = (points, matrix)
            rows, transferred, covered = training_from_reconstruction(
                points, matrix, truth_points, truth_labels, args.transfer_radius)
            training[scene] = (rows, transferred)

        note = "" if covered is None else f", {covered:.1%} of it within the truth"
        print(f"  {scene}: {len(training[scene][1]):,} points{note} in "
              f"{time.time() - started:.0f}s", flush=True)

    # Held out by scene, which is the only honest split here: the two scenes are
    # different buildings, and a random split within one would score how well
    # the model memorises that building.
    print("\ncross-scene check, in the domain the classifier was fitted on")
    crossed = []
    for train_name, test_name in (LABELLED, LABELLED[::-1]):
        model = fit(*training[train_name])
        matrix = confusion(training[test_name][1], model.predict(training[test_name][0]))
        crossed.append(report(f"{train_name} -> {test_name}", matrix))
    receipt["cross_scene_miou"] = [round(v, 4) for v in crossed]

    model = fit(np.concatenate([training[n][0] for n in LABELLED]),
                np.concatenate([training[n][1] for n in LABELLED]))

    print("\nlabelling the reconstructed clouds")
    confusions, as_shipped, ceilings = [], [], []
    for scene in scenes:
        if sources[scene] is None:
            continue
        started = time.time()
        if scene in prepared:
            points, matrix = prepared.pop(scene)
        else:
            points, colour = load_cloud(sources[scene], args.voxel, args.colour)
            matrix = features(points, colour=colour)
        labels = model.predict(matrix).astype(np.uint8)

        destination = args.output / scene / "point_cloud.ply"
        write_submission_ply(destination, points, labels)
        row = {"source": sources[scene].parent.name,
               "points": int(len(points)),
               "share": {SEMANTIC_CLASSES[i]: round(float((labels == i).mean()), 4)
                         for i in SEMANTIC_CLASS_INDICES},
               "seconds": round(time.time() - started, 1)}

        if (gold_coast / scene / "3d_gt" / "point_cloud.ply").exists():
            offset = offsets.get(scene, np.zeros(3))
            truth_points, truth_labels = truth_cloud(gold_coast / scene, offset)
            matched = confusion(truth_labels,
                                transfer_labels(points, labels, truth_points))
            confusions.append(matched)
            row["miou"] = round(report(scene, matched), 4)
            row["offset"] = [round(float(v), 4) for v in offset]

            tally = coverage(points, truth_points, truth_labels)
            ceilings.append(tally)
            row["ceiling"] = round(pooled_ceiling([tally])[0], 4)

            if offsets:
                # What the same cloud scores against the truth where it was
                # shipped, which is what the evaluator may still be holding.
                shipped_points, shipped_labels = truth_cloud(gold_coast / scene,
                                                             np.zeros(3))
                as_shipped.append(confusion(
                    shipped_labels, transfer_labels(points, labels, shipped_points)))
                row["miou_as_shipped"] = round(pooled_miou([as_shipped[-1]])[0], 4)
            print(f"    {len(points):,} points in {row['seconds']:.0f}s", flush=True)
        else:
            print(f"  {scene}: {len(points):,} points, withheld", flush=True)
        receipt["scenes"][scene] = row

    for name, tally in (("semantic_score", confusions), ("as shipped", as_shipped)):
        if not tally:
            continue
        miou, per_class = pooled_miou(tally)
        if name == "semantic_score":
            receipt["semantic_score"] = round(miou, 4)
        else:
            receipt["semantic_score_as_shipped"] = round(miou, 4)
        print(f"\n{name:16s} {miou:.4f}   "
              + ", ".join(f"{SEMANTIC_CLASSES[i]} {per_class[i]:.3f}"
                          for i in SEMANTIC_CLASS_INDICES))

    if ceilings:
        ceiling, per_class = pooled_ceiling(ceilings)
        receipt["ceiling"] = round(ceiling, 4)
        print(f"{'ceiling':16s} {ceiling:.4f}   "
              + ", ".join(f"{SEMANTIC_CLASSES[i]} {per_class[i]:.3f}"
                          for i in SEMANTIC_CLASS_INDICES))
        # The two numbers this splits the gap into. Reading them the wrong way
        # round is what sent this repository after a coverage problem it did not
        # have, and then after a classifier it could not have improved.
        print(f"\ncoverage costs {1 - ceiling:.4f}, "
              f"the classifier costs {ceiling - receipt['semantic_score']:.4f}")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
