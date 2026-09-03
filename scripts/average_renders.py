#!/usr/bin/env python3
"""Average the renders of several seeds, because one rendering run is not a result.

Training the same scene twice with the same settings moves the rendering term by
up to 0.066 - see "Rendering iterations: a wash" and the seed table in
HANDOVER.md - and the seven scenes the final phase scores are blind, so there is
no way to tell there which draw was got. Averaging removes the choice instead of
making it.

It is not only variance reduction. On the four development scenes the mean of
three seeds beats the *best* of the three, not merely their expectation, because
the seeds disagree mostly where each is separately wrong.

Every frame must be present in every input directory or the run stops. A mean
taken over whichever seeds happened to finish is a different estimator per
frame, and the one frame that silently averaged two seeds instead of three is
not something a receipt would make obvious later.

    python scripts/average_renders.py \\
        --renders ~/EditReadyGS_runs/twinworld/render_seed{0,1,2} \\
        --output ~/EditReadyGS_runs/twinworld/render_mean3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def frames_under(root: Path) -> dict[str, Path]:
    """{'<dataset>_<scene>/<stem>': path} for every render below `root`."""
    return {f"{path.parent.parent.name}/{path.stem}": path
            for path in sorted(root.glob("*/rgb/*.png"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renders", required=True, type=Path, nargs="+",
                        help="two or more directories of <dataset>_<scene>/rgb/<stem>.png")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if len(args.renders) < 2:
        raise SystemExit("averaging needs at least two render directories")

    inputs = [frames_under(root) for root in args.renders]
    for root, found in zip(args.renders, inputs):
        if not found:
            raise SystemExit(f"{root} contains no <scene>/rgb/*.png")

    keys = set(inputs[0])
    for root, found in zip(args.renders[1:], inputs[1:]):
        missing = keys ^ set(found)
        if missing:
            listed = "\n    ".join(sorted(missing)[:10])
            raise SystemExit(
                f"{root} does not hold the same frames as {args.renders[0]} - "
                f"{len(missing)} differ, so the mean would be over a different number "
                f"of seeds per frame:\n    {listed}")

    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"inputs": [str(root.resolve()) for root in args.renders], "frames": {}}
    for key in sorted(keys):
        stack = np.stack([np.asarray(Image.open(found[key]).convert("RGB"), dtype=np.float32)
                          for found in inputs])
        if len({frame.shape for frame in stack}) != 1:
            raise SystemExit(f"{key} is not the same size in every input")
        averaged = stack.mean(axis=0).round().clip(0, 255).astype(np.uint8)

        scene, stem = key.split("/")
        destination = args.output / scene / "rgb" / f"{stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(averaged).save(destination)
        receipt["frames"][key] = {"seeds": len(inputs),
                                  "spread": round(float(stack.std(axis=0).mean()), 3)}

    spreads = [row["spread"] for row in receipt["frames"].values()]
    print(f"averaged {len(keys)} frames over {len(inputs)} renders into {args.output}")
    print(f"mean per-pixel disagreement between seeds: {np.mean(spreads):.2f} of 255 "
          f"(worst frame {max(spreads):.2f})")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
