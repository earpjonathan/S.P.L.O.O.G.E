#!/bin/bash
# eval_run.sh <ply> <tag> [max_splats]
# Convert a Brush PLY into the BASELINE viewer frame and measure the ground.
#
# The up axis is NOT the Brush "Vertical axis" comment, and not that comment
# with Z negated either -- on this scene that put 94.2% of cameras UNDER the
# terrain. It is the full negation. Verified by the camera-above-terrain test
# and by the rebuilt camera frame matching viewer/aug2_cameras.json to 6e-6.
set -e
P=~/Desktop/fpv-splat
PLY=$1; TAG=$2; CAP=${3:-3000000}
AUG2_UP="-0.104898565,0.56495816,0.8184245"
OUT=$P/work/eval_$TAG.splat
CAMS=$P/work/eval_$TAG.cameras.json

$P/.venv/bin/python $P/scripts/ply2splat.py "$PLY" -o "$OUT" \
  --up "$AUG2_UP" --colmap $P/colmap/aug2/sparse/0 \
  --max-splats "$CAP" --cameras-out "$CAMS" | tail -3

$P/.venv/bin/python - "$CAMS" <<'PY'
import json, sys, numpy as np
a=np.array([c['position'] for c in json.load(open(sys.argv[1]))])
b=np.array([c['position'] for c in json.load(open('/Users/jonathanearp/Desktop/fpv-splat/viewer/aug2_cameras.json'))])
d=np.abs(a-b).max()
print(f"  frame check vs baseline: max |diff| {d:.2e} " + ("OK" if d<1e-3 else "*** MISMATCH ***"))
assert d < 1e-3, "coordinate frames differ -- comparison would be meaningless"
PY

nice -n 10 $P/.venv/bin/python $P/scripts/ground_thick.py "$OUT" \
  $P/viewer/aug2_cameras.json "$TAG"
