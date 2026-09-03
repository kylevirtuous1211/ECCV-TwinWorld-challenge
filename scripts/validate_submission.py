#!/usr/bin/env python3
"""Check a submission zip before it costs one of the ten final-phase slots.

Exit status is the alarm channel: 0 when the zip is safe to upload, 1 when the
scorer would reject or silently under-score it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinworld.submission import validate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="the submission zip to check")
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="TwinWorld_Datasets directory, holding Data_TUM and Data_Goldcoast")
    args = parser.parse_args()

    report = validate(args.zip_path.resolve(strict=True), args.dataset_root.resolve(strict=True))
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
