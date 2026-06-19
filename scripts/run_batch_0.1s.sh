#!/usr/bin/env bash
# Run full 0.1s pipeline: segment -> validate -> render
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs_0.1s
PY=.venv/bin/python

echo "=== Step 1: Segment all videos (0.1s interval) ==="
$PY scripts/segment_all.py --output-dir "$OUT"

echo "=== Step 2: Validate COCO + merge ==="
$PY scripts/validate_coco.py --output-dir "$OUT" --merge

echo "=== Step 3: Render annotated videos ==="
$PY scripts/render_annotated_videos.py --output-dir "$OUT" --skip-existing

echo "=== Done ==="
