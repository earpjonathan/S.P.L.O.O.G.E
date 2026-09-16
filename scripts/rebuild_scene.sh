#!/bin/bash
# Rebuild one scene's viewer assets on the same terms as merged:
# 3M splats, NO opacity boost (the boost was a workaround for the znear bug),
# up derived from ground geometry and validated, cameras + trajectory regenerated.
#   usage: rebuild_scene.sh <colmap-model> <ply> <prefix>
set -e
P=~/Desktop/fpv-splat
cd $P
BEST="$1"; PLY="$2"; PRE="$3"
echo "=== rebuild $PRE  $(date +%T) ==="
echo "  model $BEST"
echo "  ply   $PLY"

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
print(','.join(f'{v:.9f}' for v in n))
PY
)
echo "  derived up: $UP"
echo "--- up validation (cameras must sit ABOVE terrain) ---"
$P/.venv/bin/python scripts/check_up.py "$PLY" "$BEST" derived="$UP"

$P/.venv/bin/python scripts/ply2splat.py "$PLY" -o $P/viewer/${PRE}_web.splat \
  --up "$UP" --colmap "$BEST" --max-splats 3000000 \
  --cameras-out $P/viewer/${PRE}_cameras.json

$P/.venv/bin/python scripts/trajectory.py --colmap "$BEST" \
  --cameras $P/viewer/${PRE}_cameras.json -o $P/viewer/${PRE}_traj.json

ls -la $P/viewer/${PRE}_*
echo "=== rebuild $PRE DONE $(date +%T) ==="
