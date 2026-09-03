#!/usr/bin/env python3
"""Move named clouds inside a built submission, and change nothing else.

The last four archives differ only in what `scene_011` and `scene_012` are
shifted by, and each was rebuilt from the clouds on the cluster to change two
numbers. Rebuilding is the wrong tool: it puts every other entry back through the
pipeline, so "the rest of the archive is unchanged" becomes a claim rather than a
fact, and a slot is spent on an archive whose other fifty entries were not the
ones that scored.

This edits the archive in place instead. It rewrites only the named clouds, by
adding a displacement to their coordinates, and copies every other entry through.
Afterwards it hashes the *decompressed* content of all fifty-two entries in both
archives and reports how many differ, so the one-variable claim is checked rather
than asserted.

    python scripts/reshift_submission.py --submission <zip> \\
        --shift gold_coast/scene_011 0.0165 0.5126 0 \\
        --shift gold_coast/scene_012 0.0165 0.5126 0 \\
        --output <zip>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def move(payload: bytes, delta: np.ndarray) -> tuple[bytes, int]:
    ply = PlyData.read(io.BytesIO(payload))
    vertex = ply["vertex"].data.copy()
    for axis, value in zip(("x", "y", "z"), delta):
        if value:
            vertex[axis] = (vertex[axis].astype(np.float64) + value).astype(vertex[axis].dtype)
    out = io.BytesIO()
    PlyData([PlyElement.describe(vertex, "vertex")],
            text=False, byte_order="<").write(out)
    return out.getvalue(), len(vertex)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--shift", action="append", nargs=4, metavar=("SCENE", "DX", "DY", "DZ"),
                        default=[], help="a scene as it is named in the archive, then the "
                                         "displacement to add to its cloud, in metres")
    parser.add_argument("--replace", action="append", nargs=2, metavar=("SCENE", "PLY"),
                        default=[], help="a scene as it is named in the archive, then a cloud "
                                         "to put in its place. The replacement is written "
                                         "through the same writer as a shift, so the header "
                                         "and dtype are the archive's and not the file's")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.shift and not args.replace:
        raise SystemExit("nothing to do: pass --shift or --replace")
    deltas = {scene: np.array([float(dx), float(dy), float(dz)])
              for scene, dx, dy, dz in args.shift}
    targets = {f"{scene}/3D_point_cloud/point_cloud.ply": delta
               for scene, delta in deltas.items()}
    swaps = {f"{scene}/3D_point_cloud/point_cloud.ply": Path(path)
             for scene, path in args.replace}
    overlap = set(targets) & set(swaps)
    if overlap:
        raise SystemExit(f"both shifted and replaced, which is ambiguous: {sorted(overlap)}")

    source = zipfile.ZipFile(args.submission)
    names = source.namelist()
    missing = [name for name in list(targets) + list(swaps) if name not in names]
    if missing:
        raise SystemExit(f"not in the archive: {missing}")

    receipt = {"source": str(args.submission), "output": str(args.output),
               "shifts": {k: [float(v) for v in d] for k, d in deltas.items()},
               "replacements": {k: str(v) for k, v in swaps.items()},
               "entries": len(names), "moved": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as out:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename in targets:
                payload, count = move(payload, targets[info.filename])
                receipt["moved"][info.filename] = {"points": count}
                print(f"  moved {info.filename} by "
                      f"{np.round(targets[info.filename], 4)}  ({count:,} points)", flush=True)
            elif info.filename in swaps:
                before = len(PlyData.read(io.BytesIO(payload))["vertex"].data)
                payload, count = move(swaps[info.filename].read_bytes(), np.zeros(3))
                receipt["moved"][info.filename] = {"points": count, "points_before": before,
                                                   "from": str(swaps[info.filename])}
                print(f"  replaced {info.filename} with {swaps[info.filename]}  "
                      f"({before:,} -> {count:,} points)", flush=True)
            out.writestr(zipfile.ZipInfo(info.filename, date_time=info.date_time),
                         payload, compress_type=info.compress_type)

    # The claim this archive rests on is that nothing but those clouds moved.
    # Hash what the scorer reads - the decompressed bytes - and count.
    written = zipfile.ZipFile(args.output)
    if set(written.namelist()) != set(names):
        raise SystemExit("the output does not hold the same set of entries")
    differ = [name for name in names
              if hashlib.sha256(source.read(name)).digest()
              != hashlib.sha256(written.read(name)).digest()]
    receipt["entries_differing"] = sorted(differ)
    print(f"\n{len(names) - len(differ)} of {len(names)} entries are byte-identical to the source")
    print(f"the {len(differ)} that differ: {sorted(differ)}")
    receipt["megabytes"] = round(args.output.stat().st_size / 1e6, 1)
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    if sorted(differ) != sorted(list(targets) + list(swaps)):
        raise SystemExit("something other than the named clouds changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
