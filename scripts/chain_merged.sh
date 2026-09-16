#!/bin/bash
# Post-COLMAP chain for the merged hilltop+cemetery scene.
#
# Waits for the merged mapper, validates the model, trains, then builds the
# viewer assets. Every stage is gated: a bad model aborts loudly rather than
# burning 4h of GPU on garbage.
#
# Waits on any running brush-cli too, so it never competes with a distortion
# sweep for the GPU.
set -e
P=~/Desktop/fpv-splat
cd $P
LOG=$P/work/chain_merged.log
exec > >(tee -a "$LOG") 2>&1

echo "=== chain_merged start $(date +%T) ==="

echo "--- waiting for merged mapper ---"
while pgrep -f "colmap mapper.*merged/database" > /dev/null; do sleep 120; done
echo "--- mapper exited $(date +%T) ---"

# --- pick the largest model -------------------------------------------------
BEST=""; BESTN=0
for m in $P/colmap/merged/sparse/*/; do
  [ -f "$m/images.bin" ] || continue
  n=$(colmap model_analyzer --path "$m" 2>&1 | grep -m1 "Registered images" | grep -oE "[0-9]+$" || echo 0)
  echo "  model $m -> $n registered"
  if [ "${n:-0}" -gt "$BESTN" ]; then BESTN=$n; BEST=$m; fi
done
if [ -z "$BEST" ]; then
  echo "ABORT: no model written to colmap/merged/sparse -- falling back to newest snapshot"
  BEST=$(ls -dt $P/colmap/merged/snapshots/*/ 2>/dev/null | head -1)
  [ -z "$BEST" ] && { echo "ABORT: no snapshot either"; exit 1; }
  BESTN=$(colmap model_analyzer --path "$BEST" 2>&1 | grep -m1 "Registered images" | grep -oE "[0-9]+$")
  echo "  using snapshot $BEST ($BESTN registered)"
fi
echo "=== chosen model: $BEST  ($BESTN registered) ==="

# --- gate: did the two sessions actually fuse? ------------------------------
# Cemetery frames are 0057-0060, hilltop 0050-0055. One model containing both
# is the whole point; a model with only one session is a FAILED merge and must
# not be silently trained as if it were the merged scene.
$P/.venv/bin/python - "$BEST" "$BESTN" <<'PY'
import sys, os, re
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from colmap_io import read_images
im = read_images(f'{sys.argv[1]}/images.bin')
clips = {}
for v in im.values():
    m = re.match(r'(\d{4})', v['name'])
    if m: clips[m.group(1)] = clips.get(m.group(1), 0) + 1
hill = sum(n for c, n in clips.items() if c in ('0050','0051','0052','0053','0054','0055'))
cem  = sum(n for c, n in clips.items() if c in ('0057','0058','0059','0060'))
print('  per-clip frame counts:', dict(sorted(clips.items())))
print(f'  hilltop {hill}   cemetery {cem}')
assert hill > 200 and cem > 200, \
    f'MERGE FAILED: model has hilltop={hill} cemetery={cem}; not a fused scene'
print('  MERGE CONFIRMED: both sessions present in one model')
PY

colmap model_analyzer --path "$BEST" 2>&1 | grep -E "Registered images|Points:|track length|reprojection"

# --- wait for the GPU -------------------------------------------------------
# -x matches the process NAME, not the full command line. A -f substring
# match here would also match any shell whose command line merely CONTAINS
# "brush-cli", which is how a previous pkill took out five waiter shells.
echo "--- waiting for any running brush-cli ---"
while pgrep -x brush-cli > /dev/null; do sleep 120; done
echo "--- GPU free $(date +%T) ---"

# --- train ------------------------------------------------------------------
# 8M cap to match the aug2 baseline, so merged numbers are comparable.
REL=${BEST#$P/}
CACHE=12GiB bash scripts/run_training.sh "${REL%/}" 30000 8000000 merged_out merged merged_train

PLY=$P/merged_out/site_30000.ply
[ -f "$PLY" ] || { echo "ABORT: no PLY at $PLY"; exit 1; }

# --- derive scene up from geometry, then VALIDATE it ------------------------
# Never trust Brush's `Vertical axis` comment: on this project it has been
# wrong in both a sign and a full-negation sense, and a wrong up puts the
# cameras underground while still looking plausible.
UP=$($P/.venv/bin/python - "$BEST" <<'PY'
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from colmap_io import read_points3D, read_images, qvec2R
from find_up import ground_normal
sp = sys.argv[1]
im = read_images(f'{sp}/images.bin')
C = np.array([-qvec2R(x['q']).T @ x['t'] for x in im.values()])
_, P0, _, _, _ = read_points3D(f'{sp}/points3D.bin')
n, frac = ground_normal(P0, C)
print(','.join(f'{v:.9f}' for v in n), file=sys.stderr)
print(','.join(f'{v:.9f}' for v in n))
PY
)
echo "=== derived up: $UP ==="

echo "--- up-axis validation (cameras must sit ABOVE terrain) ---"
$P/.venv/bin/python scripts/check_up.py "$PLY" "$BEST" derived="$UP"

# --- build viewer assets ----------------------------------------------------
# K=12 over a WIDE band. The default -0.6..0.15 band was tuned on the
# single-site scene and catches only 24% of merged splats vs 45% at -1.5..0.4 --
# on this terrain it left most ground splats unboosted. Widening more than
# doubles the benefit at the worst poses (46.9% -> 35.9% holes). Also build an
# unboosted copy for equal-terms comparison against the aug2 baseline.
$P/.venv/bin/python scripts/ply2splat.py "$PLY" -o $P/viewer/merged_web.splat \
  --up "$UP" --colmap "$BEST" --max-splats 3000000 \
  --ground-opacity-boost 12 --ground-band -1.5 0.4 \
  --cameras-out $P/viewer/merged_cameras.json

$P/.venv/bin/python scripts/ply2splat.py "$PLY" -o $P/work/merged_raw.splat \
  --up "$UP" --colmap "$BEST" --max-splats 3000000

$P/.venv/bin/python scripts/trajectory.py --colmap "$BEST" \
  --cameras $P/viewer/merged_cameras.json -o $P/viewer/merged_traj.json

ls -la $P/viewer/merged_*
echo "=== chain_merged DONE $(date +%T) ==="
echo "NOTE: viewer assets built but NOT deployed to work/webroot -- deploy is"
echo "      a publish action, left for explicit approval."
