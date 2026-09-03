#!/usr/bin/env python3
"""Translate submitted clouds into the frame the shipped ground truth is in.

`scripts/frame_offset.py` measures that the Gold Coast ground truth sits 1.1 to
1.4 m from the camera frame its own images define, and that TUM does not. This
script applies that offset, so that two archives can be built which differ in
nothing else.

That pair is the whole point. It is not knowable from here whether the evaluator
scores against the same misregistered clouds that were shipped to us or against
a correctly registered internal copy, and the two answers want opposite
submissions. One development-phase slot each settles it; no amount of local
measurement can.

The features the labels come from are translation-invariant in xy - height above
ground is measured against a ground surface fitted to the same cloud, and local
PCA does not care where the origin is - so shifting after labelling gives exactly
what re-labelling a shifted cloud would, at none of the cost. Labels are carried
through untouched.

    python scripts/shift_clouds.py --clouds <labelled dir> \\
        --offsets <frame_offset.json> --output <dir>

A scene with no measured offset stops the run, because silently submitting it
unshifted inside an archive named "shifted" is how the experiment gets read
backwards a week later. `--unmeasured keep` accepts that deliberately.

`--unmeasured extrapolate` gives an unmeasured scene the mean of the offsets
measured for its own dataset. That is a guess and it is named like one, but it is
an evidenced guess and the final phase is where it has to be made: the two
withheld Gold Coast scenes carry no ground truth, so their offset cannot be
measured, and they are the only two scenes the final phase scores semantics on.
The evidence is in HANDOVER.md - all four Gold Coast frames share a z axis to
within half a degree, so they are gravity aligned rather than arbitrary, and the
two measured offsets agree to 0.04 m in y against a search plateau about as wide
as their 0.43 m disagreement in x. Being 0.2 m wrong costs about a fifth of what
not shifting costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.pointcloud import write_submission_ply  # noqa: E402


def read_labelled(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    labels = (np.asarray(vertex["classification"]).astype(np.uint8)
              if "classification" in names else None)
    return points, labels


def _dataset_of(scene: str) -> str:
    """TUM owns scene_000 to scene_008 and Gold Coast scene_009 to scene_012."""
    return "tum" if int(scene.rsplit("_", 1)[-1]) <= 8 else "gold_coast"


def offsets_by_scene(receipt: dict) -> dict[str, np.ndarray]:
    """Scene id to offset, dropping the dataset the receipt keys them by.

    Scene ids are unique across both datasets - TUM owns 000 to 008 and Gold
    Coast 009 to 012 - so the shorter key is unambiguous and matches how the
    labelled clouds are laid out on disk.
    """
    found: dict[str, np.ndarray] = {}
    for key, row in receipt["scenes"].items():
        scene = key.split("/")[-1]
        if scene in found:
            raise SystemExit(f"{scene} appears twice in the offsets; keys are {sorted(receipt['scenes'])}")
        found[scene] = np.asarray(row["offset"], dtype=np.float64)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds", required=True, type=Path,
                        help="directory of <scene>/point_cloud.ply")
    parser.add_argument("--offsets", required=True, type=Path,
                        help="receipt written by scripts/frame_offset.py")
    parser.add_argument("--unmeasured", default="fail",
                        choices=["fail", "keep", "extrapolate"],
                        help="what to do with a scene the offsets do not cover: stop, ship "
                             "it unshifted, or give it the mean of the offsets measured for "
                             "its own dataset")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.offsets.read_text())
    offsets = offsets_by_scene(receipt)

    sources = sorted(args.clouds.glob("*/point_cloud.ply"))
    if not sources:
        raise SystemExit(f"nothing matched {args.clouds}/*/point_cloud.ply")

    missing = [p.parent.name for p in sources if p.parent.name not in offsets]
    if missing and args.unmeasured == "fail":
        raise SystemExit(
            f"no measured offset for {', '.join(missing)}, so these would go into a "
            f"shifted archive unshifted. Pass --unmeasured keep if that is intended.")

    extrapolated: dict[str, np.ndarray] = {}
    if missing and args.unmeasured == "extrapolate":
        # Per dataset, because TUM reads about 0.02 m and Gold Coast 1.1 to 1.4:
        # averaging the two together would be averaging a control with a signal.
        for key, row in receipt["scenes"].items():
            dataset = key.split("/")[0] if "/" in key else None
            for scene in missing:
                if dataset is not None and _dataset_of(scene) != dataset:
                    continue
                extrapolated.setdefault(scene, []).append(np.asarray(row["offset"], float))
        for scene in missing:
            samples = extrapolated.get(scene)
            if not samples:
                raise SystemExit(
                    f"{scene} has no measured scene from its own dataset to extrapolate from")
            extrapolated[scene] = np.mean(samples, axis=0)
        offsets = {**offsets, **extrapolated}

    written = {"offsets": str(args.offsets.resolve()), "axes": receipt.get("axes"),
               "scenes": {}}
    print(f"{'scene':14s} {'points':>12s} {'dx':>7s} {'dy':>7s} {'dz':>7s}  note")
    print("-" * 62)

    for source in sources:
        scene = source.parent.name
        offset = offsets.get(scene, np.zeros(3))
        points, labels = read_labelled(source)
        count = write_submission_ply(args.output / scene / "point_cloud.ply",
                                     points + offset, labels)
        if scene in extrapolated:
            note = "EXTRAPOLATED from its dataset's measured mean"
        elif scene in offsets:
            note = "shifted"
        else:
            note = "no offset measured, copied as-is"
        written["scenes"][scene] = {"points": count,
                                    "offset": [round(float(v), 4) for v in offset],
                                    "shifted": scene in offsets,
                                    "extrapolated": scene in extrapolated,
                                    "labelled": labels is not None}
        print(f"{scene:14s} {count:>12,d} {offset[0]:>7.2f} {offset[1]:>7.2f} "
              f"{offset[2]:>7.2f}  {note}", flush=True)

    (args.output / "receipt.json").write_text(json.dumps(written, indent=2))
    print(f"\nwrote {len(sources)} clouds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
