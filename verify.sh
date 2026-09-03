#!/usr/bin/env bash
# Validate a submission archive and score it on the six released scenes.
#
#     bash verify.sh <archive.zip> <TwinWorld_Datasets>
#
# Uses `python` from the active environment; set PYTHON=<path> to override.
#
# The seven withheld scenes cannot be scored anywhere but on the leaderboard, so
# what this checks is the format, the point budget, and the three metrics on the
# scenes that ship ground truth.
set -euo pipefail
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
if [ -z "$PYTHON" ]; then echo "no python on PATH; set PYTHON=<path>" >&2; exit 1; fi
archive=${1:?usage: verify.sh <archive.zip> <dataset-root>}
root=${2:?usage: verify.sh <archive.zip> <dataset-root>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

echo "=== format and point budget ==="
"$PYTHON" scripts/validate_submission.py "$archive" --dataset-root "$root"

echo
echo "=== the three metrics, on the six released scenes ==="
"$PYTHON" scripts/score_local.py --dataset-root "$root" --submission "$archive"
