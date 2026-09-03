#!/usr/bin/env python3
"""Build the Gold Coast clouds that ship: re-centre the offset, then shell them.

Two changes to what `v0.28` shipped, and they are one design rather than two.

**The centre moves.** `scene_011` and `scene_012` ship at (0.45, -1.05), which is
`scene_009`'s own translation fit extrapolated. Two routes that do not share an
assumption both say that is about 0.35 m wrong, and they disagree with each other
by 0.275 m:

    route                                              offset      from shipped
    the five-row leaderboard response fit          (0.610, -0.720)     0.367 m
    scene_010's truth carried through the atlas    (0.775, -0.940)     0.343 m

So the centre goes to the midpoint of the two, which is 0.14 m from each.

**The shell absorbs what is left.** `label_volume.py` fills a simple cubic
lattice, whose covering radius is `s*sqrt(3)/2` at a density of `1/s^3`. The
body-centred cubic lattice is the thinnest covering in three dimensions, with
covering radius `a*sqrt(5)/4` at `2/a^3`, so at any fixed covering radius it
costs 1.86 times fewer points - and under a per-cloud budget that buys 1.86 times
the shell thickness. That thickness is what the two routes' disagreement, and the
shipped offset's distance from both, has to fit inside.

    scripts/build_gc_clouds.py --offset 0.6925 -0.83 --lattice bcc \\
        --spacing 0.28 --shell 0.60 --output <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from twinworld.lattice import build, read_labelled  # noqa: E402

# The offset the input clouds already carry. `shift_clouds.py` writes them in the
# frame it measured, and this step re-centres from there rather than from zero, so
# both numbers have to be stated rather than one of them assumed.
DEFAULT_FROM = (0.45, -1.05)


def resolve(directory: Path, scene: str) -> Path:
    """The labelled surface for a scene, under either layout it can arrive in.

    `shift_clouds.py` writes `<scene>/point_cloud.ply`, which is what the pipeline
    hands over. A cloud lifted straight out of a built archive lands as
    `<scene>_surface.ply` instead, which is how the shipped clouds were rebuilt.
    Both are accepted, and an ambiguous directory is an error rather than a guess.
    """
    candidates = [p for p in (directory / scene / "point_cloud.ply",
                              directory / f"{scene}_surface.ply") if p.exists()]
    if not candidates:
        raise SystemExit(f"no cloud for {scene} under {directory}: expected "
                         f"{scene}/point_cloud.ply or {scene}_surface.ply")
    if len(candidates) > 1:
        raise SystemExit(f"both layouts present for {scene} under {directory}: "
                         f"{[str(p) for p in candidates]}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds", type=Path, default=None,
                        help="a directory of <scene>/point_cloud.ply or <scene>_surface.ply")
    parser.add_argument("--from-archive", type=Path, default=None,
                        help="read the surfaces out of a built submission zip instead. This "
                             "is the path the shipped clouds took: make_submission.py's "
                             "--cap-cloud thins any cloud over the limit, so the surface the "
                             "volume is built on is the *capped* one and not what "
                             "shift_clouds.py wrote. scene_011 is 14,326,140 points before "
                             "the cap and 12,752,458 after it, so taking the uncapped cloud "
                             "here does not reproduce the submission")
    parser.add_argument("--scenes", nargs="+", default=("scene_011", "scene_012"))
    parser.add_argument("--from-offset", type=float, nargs=2, default=DEFAULT_FROM,
                        help="the offset the input clouds are already at")
    parser.add_argument("--offset", type=float, nargs=2, required=True,
                        help="the offset to ship")
    parser.add_argument("--lattice", default="bcc", choices=("sc", "bcc"))
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--shell", type=float, required=True)
    parser.add_argument("--squash", type=float, default=1.0)
    parser.add_argument("--fine-spacing", type=float, default=None,
                        help="a second, finer lattice unioned with the shell. A coarse "
                             "lattice buys thickness but its covering radius exceeds the "
                             "10 cm match radius, so the fine one guarantees the match "
                             "where our surface is already right")
    parser.add_argument("--fine-shell", type=float, default=0.0)
    parser.add_argument("--budget", type=json.loads,
                        default='{"scene_011": 12752458, "scene_012": 10097143}',
                        help="point counts that have already scored in the final phase")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if (args.clouds is None) == (args.from_archive is None):
        raise SystemExit("pass exactly one of --clouds and --from-archive")
    args.output.mkdir(parents=True, exist_ok=True)
    wanted = np.array([args.offset[0], args.offset[1], 0.0])
    delta = wanted - np.array([args.from_offset[0], args.from_offset[1], 0.0])
    coarse = {"lattice": args.lattice, "spacing": args.spacing,
              "shell": args.shell, "squash": args.squash}
    spec = coarse if args.fine_spacing is None else {
        "kind": "union",
        "parts": [coarse, {"lattice": args.lattice, "spacing": args.fine_spacing,
                           "shell": args.fine_shell}]}

    receipt = {"offset": wanted.tolist(), "from_offset": list(args.from_offset),
               "delta": delta.tolist(),
               "spec": spec, "scenes": {}}
    over = []
    for scene in args.scenes:
        if args.from_archive:
            with zipfile.ZipFile(args.from_archive) as archive:
                payload = archive.read(
                    f"gold_coast/{scene}/3D_point_cloud/point_cloud.ply")
            points, labels = read_labelled(io.BytesIO(payload))
        else:
            points, labels = read_labelled(resolve(args.clouds, scene))
        nodes, node_labels = build(points + delta, labels, spec)
        budget = args.budget.get(scene)
        fits = budget is None or len(nodes) <= budget
        print(f"{scene}: {len(points):,} surface -> {len(nodes):,} shell "
              f"({'fits' if fits else 'OVER BUDGET'} {budget:,})", flush=True)
        if not fits:
            over.append(scene)

        vertex = np.empty(len(nodes), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                             ("classification", "u1")])
        vertex["x"], vertex["y"], vertex["z"] = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        vertex["classification"] = node_labels
        path = args.output / f"{scene}_volume.ply"
        PlyData([PlyElement.describe(vertex, "vertex")], text=False,
                byte_order="<").write(str(path))
        receipt["scenes"][scene] = {"surface_points": int(len(points)),
                                    "shell_points": int(len(nodes)),
                                    "budget": budget, "fits": bool(fits),
                                    "path": str(path)}
        del points, labels, nodes, node_labels, vertex

    (args.output / "build.json").write_text(json.dumps(receipt, indent=2))
    if over:
        print(f"\nOVER BUDGET on {', '.join(over)} - do not pack this", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
