#!/usr/bin/env python3
"""Assemble the real submission from the artefacts three other scripts produced.

Until now the only complete zip this repository could build was the baseline,
which renders the nearest train image and submits the sparse COLMAP cloud. The
three real terms were each measured on their own - renders from
`train_scene.py`, surface clouds from `export_geometry.py`, labelled clouds from
`label_cloud.py` - and their scores were added by hand. Added by hand is not a
submission: a term can be right in isolation and still be missing, misnamed or
the wrong variant once thirteen scenes have to appear in one archive.

So this script does the joining, and refuses to guess. Every slot is resolved to
exactly one file on disk or the run stops. In particular a glob that matches
more than one export is an error rather than a first-match, because the exports
directory accumulates variants that differ only in a suffix, and picking the
alphabetically first one is how a sweep's throwaway run ends up in a submission.

    python scripts/make_submission.py \
        --dataset-root ~/EditReadyGS_runs/twinworld/TwinWorld_Datasets \
        --renders ~/EditReadyGS_runs/twinworld/final/render \
        --clouds ~/EditReadyGS_runs/twinworld/geom \
        --tum-variant n05d001 --gold-coast-variant d0.01 \
        --semantic ~/EditReadyGS_runs/twinworld/final/semantic \
        --output ~/EditReadyGS_runs/twinworld/final/submission.zip

It validates what it wrote before exiting, so a zip that reaches the end of this
script has already passed `validate_submission.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.dataset import Scene, load_scenes  # noqa: E402
from twinworld.pointcloud import read_submission_ply, thin_unscored_ply  # noqa: E402
from twinworld.submission import validate  # noqa: E402

# The exports are named by dataset with a shorter prefix than the submission
# uses, and that prefix is what `export_geometry.py` output directories carry.
CLOUD_PREFIX = {"tum": "tum", "gold_coast": "gc"}

# PNG is already deflated internally, so compressing it again buys nothing and
# costs a pass over every pixel. The clouds are raw float32 and do shrink, but
# at level 1: the levels above it spend minutes on gigabytes for a few percent.
PLY_COMPRESS_LEVEL = 1


class Missing(RuntimeError):
    """A submission slot could not be resolved to exactly one file."""


def resolve_one(candidates: list[Path], what: str, searched: str) -> Path:
    if not candidates:
        raise Missing(f"{what}: nothing matched {searched}")
    if len(candidates) > 1:
        listed = "\n    ".join(str(c) for c in sorted(candidates))
        raise Missing(
            f"{what}: {len(candidates)} files matched {searched}, so which one belongs "
            f"in the submission is undecided. Name the variant explicitly.\n    {listed}")
    return candidates[0]


def render_path(renders: Path, scene: Scene, stem: str) -> Path:
    pattern = f"{scene.dataset}_{scene.scene_id}/rgb/{stem}.png"
    return resolve_one([p for p in [renders / pattern] if p.exists()],
                       f"{scene.dataset}/{scene.scene_id} render {stem}",
                       str(renders / pattern))


def cloud_path(clouds: Path, semantic: Path | None, scene: Scene,
               variants: dict[str, str]) -> Path:
    if scene.needs_classification and semantic is not None:
        pattern = f"{scene.scene_id}/point_cloud.ply"
        return resolve_one([p for p in [semantic / pattern] if p.exists()],
                           f"{scene.dataset}/{scene.scene_id} labelled cloud",
                           str(semantic / pattern))
    variant = variants.get(scene.dataset) or "*"
    pattern = f"{CLOUD_PREFIX[scene.dataset]}_{scene.scene_id}_{variant}/point_cloud.ply"
    return resolve_one(sorted(clouds.glob(pattern)),
                       f"{scene.dataset}/{scene.scene_id} cloud",
                       str(clouds / pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--renders", required=True, type=Path,
                        help="directory of <dataset>_<scene>/rgb/<stem>.png")
    parser.add_argument("--clouds", required=True, type=Path,
                        help="directory of <tum|gc>_<scene>_<variant>/point_cloud.ply")
    parser.add_argument("--semantic", type=Path, default=None,
                        help="directory of <scene>/point_cloud.ply labelled clouds; "
                             "these replace --clouds for Gold Coast")
    parser.add_argument("--tum-variant", default=None,
                        help="the export suffix to take for TUM, e.g. n05d001")
    parser.add_argument("--gold-coast-variant", default=None,
                        help="the export suffix to take for Gold Coast, e.g. d0.01")
    parser.add_argument("--cap-cloud", type=int, default=None, metavar="POINTS",
                        help="thin any cloud above this many points until it is under. The "
                             "scorer is killed by a single cloud above some size - 13.84M "
                             "has scored and 15.11M was killed with a null exit code and no "
                             "log - and it is the largest cloud that decides, not the "
                             "archive: 1457 MB has scored where 901 MB was killed.")
    parser.add_argument("--thin-unscored", type=float, default=None, metavar="VOXEL",
                        help="thin the clouds of the scenes this phase does not score to "
                             "this spacing in metres. A phase scores exactly the scenes "
                             "its reference data holds, and the final phase's holds the "
                             "seven withheld ones, so the six that ship ground truth are "
                             "evaluator load that cannot score. They still have to be "
                             "present and valid, which is why they are thinned and not "
                             "dropped.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scenes = load_scenes(args.dataset_root)
    variants = {"tum": args.tum_variant, "gold_coast": args.gold_coast_variant}

    # Resolve everything before writing anything. A zip that turns out to be
    # incomplete halfway through is worse than no zip: it is on disk, it looks
    # finished, and the scorer is the one that finds out.
    plan: list[tuple[Scene, list[tuple[str, Path]]]] = []
    problems: list[str] = []
    for scene in scenes:
        entries: list[tuple[str, Path]] = []
        prefix = f"{scene.dataset}/{scene.scene_id}"
        for frame in scene.frames:
            try:
                entries.append((f"{prefix}/rgb/{frame.stem}.png",
                                render_path(args.renders, scene, frame.stem)))
            except Missing as error:
                problems.append(str(error))
        try:
            entries.append((f"{prefix}/3D_point_cloud/point_cloud.ply",
                            cloud_path(args.clouds, args.semantic, scene, variants)))
        except Missing as error:
            problems.append(str(error))
        plan.append((scene, entries))

    if problems:
        print(f"cannot assemble a complete submission - {len(problems)} slot(s) unresolved:\n",
              file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict = {"output": str(args.output), "scenes": {}}

    # Thin before the zip is opened, so a cloud that will not read stops the run
    # while there is still nothing on disk pretending to be a submission.
    staging = tempfile.TemporaryDirectory(prefix="twinworld-thin-")
    resolved = {f"{scene.dataset}/{scene.scene_id}": entries[-1][1] for scene, entries in plan}
    if args.cap_cloud is not None:
        for scene, entries in plan:
            name, source = entries[-1]
            points, _ = read_submission_ply(source)
            if len(points) <= args.cap_cloud:
                continue
            destination = Path(staging.name) / "capped" / scene.dataset / scene.scene_id / "point_cloud.ply"
            # Walk the voxel up rather than solving for it: the relation between
            # spacing and survivors depends on the surface, and one pass per step
            # is cheaper than being clever about a cloud that is read once.
            voxel = 0.02
            while True:
                voxel = round(voxel * 1.1, 4)
                before, after = thin_unscored_ply(source, destination, voxel)
                if after <= args.cap_cloud:
                    break
            entries[-1] = (name, destination)
            print(f"capped {scene.dataset}/{scene.scene_id}: {before:,d} -> {after:,d} points "
                  f"at {voxel} m", flush=True)
        print()

    if args.thin_unscored is not None:
        for scene, entries in plan:
            if scene.is_withheld:
                continue
            name, source = entries[-1]
            destination = Path(staging.name) / scene.dataset / scene.scene_id / "point_cloud.ply"
            before, after = thin_unscored_ply(source, destination, args.thin_unscored)
            entries[-1] = (name, destination)
            print(f"thinned {scene.dataset}/{scene.scene_id} at {args.thin_unscored} m: "
                  f"{before:,d} -> {after:,d} points ({after / before:.1%})", flush=True)
        print()

    print(f"{'scene':22s} {'renders':>8s} {'cloud MB':>9s}  source")
    print("-" * 96)

    with zipfile.ZipFile(args.output, "w") as archive:
        for scene, entries in plan:
            key = f"{scene.dataset}/{scene.scene_id}"
            cloud_source = resolved[key]
            packed = entries[-1][1]
            for name, source in entries:
                if name.endswith(".png"):
                    archive.write(source, name, zipfile.ZIP_STORED)
                else:
                    archive.write(source, name, zipfile.ZIP_DEFLATED,
                                  compresslevel=PLY_COMPRESS_LEVEL)
            megabytes = packed.stat().st_size / 1e6
            row = {
                "renders": len(entries) - 1,
                "render_source": str(entries[0][1].parent),
                "cloud_source": str(cloud_source.resolve()),
                "cloud_megabytes": round(megabytes, 1),
            }
            if packed != cloud_source:
                row["thinned"] = True
            receipt["scenes"][key] = row
            print(f"{scene.dataset + '/' + scene.scene_id:22s} {len(entries) - 1:>8d} "
                  f"{megabytes:>9.1f}  {cloud_source.resolve().parent.name}"
                  f"{'  (thinned)' if packed != cloud_source else ''}")

    staging.cleanup()

    size = args.output.stat().st_size / 1e6
    receipt["megabytes"] = round(size, 1)
    frames = sum(row["renders"] for row in receipt["scenes"].values())
    print(f"\nwrote {args.output} - {len(scenes)} scenes, {frames} frames, {size:.0f} MB")

    report = validate(args.output.resolve(), args.dataset_root.resolve())
    print("\n" + report.render())
    receipt["validation"] = {"ok": report.ok, "errors": report.errors,
                             "warnings": report.warnings}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
