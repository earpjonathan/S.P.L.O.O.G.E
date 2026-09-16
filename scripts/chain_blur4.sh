#!/bin/bash
# Final blur queue. Results so far, all at 20k on aug2, scored on the SAME
# 1920 yardstick (P4 downsampled to 1920 before scoring -- see below):
#
#   C0  8M                     bottom 18.2%   top 40.4%
#   P1  16M cap (took 14.5M)   bottom 18.6%   top 41.0%   ratio unmoved: capacity is not the lever
#   P2  16M + split 0.03       bottom 20.3%   top 42.1%   ratio moved in every band  <-- best
#   P4  native 2688 + split    bottom 20.2%   top 43.0%   near-field TIE with P2
#   P5  24M + split            CRASHED -- 4 GiB single-buffer ceiling, see below
#
# P4 was first credited with 19.7% because sharpness.py took the first
# alphabetical frames/* match (1920) and upscaled it to the 2688 render --
# scoring against a reference with no detail above 1920's Nyquist. Against its
# OWN 2688 GT it scores 16.1%. Downsampled to 1920 it is 20.2%, i.e. training at
# native resolution did not buy near-field detail; it just failed proportionally
# at a harder target. Resolution is not the lever either.
#
# P5 died on `failed to reserve 4336857088 bytes`: wgpu/Metal caps a SINGLE
# buffer at 4 GiB, and the per-splat buffer is ~181 B, so the hard ceiling is
# ~23.7M splats -- nothing to do with the 48 GB of system RAM. Retried here at
# 22M, which is the last open question: P2 pinned its 16M cap by iteration 10k,
# so unlike P1 it has a mechanism that actually consumes budget.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_blur.log 2>&1
echo "=== chain_blur4 $(date +%T) ==="

wait_gpu(){ while pgrep -x brush-cli > /dev/null; do sleep 60; done; }
detail(){ .venv/bin/python scripts/sharpness.py "$1" 40 2>/dev/null \
          | sed -n 's/.*BOTTOM third.*detail retained *\([0-9.]*\)%.*/\1/p'; }
splats(){ head -c 400 "$1" 2>/dev/null | sed -n 's/^element vertex //p' | head -1; }

# P3 (cropped LPIPS) is running now; wait it out and score it.
wait_gpu
if [ -d p3_out/eval_20000 ]; then
  D3=$(detail p3_out/eval_20000)
  echo "[P3] lpips crop 256   detail ${D3:-n/a}%   splats $(splats p3_out/site_20000.ply)"
  .venv/bin/python scripts/detail_stratified.py p3_out/eval_20000 40 2>/dev/null | tail -9
else
  echo "[P3] no eval_20000 -- run did not complete"; D3=""
fi

# P5 retry, under the 4 GiB buffer ceiling.
if [ ! -d p5_out/eval_20000 ]; then
  wait_gpu
  echo "--- [P5R] split 0.03, cap 22M  START $(date +%T) ---"
  rm -rf p5_out p5_out_train
  s=$(date +%s)
  EXTRA_ARGS="--growth-stop-iter 20000 --split-at-screen-size 0.03" MAXRES=1920 CACHE=8GiB \
    bash scripts/run_training2.sh colmap/aug2/sparse/0 20000 22000000 \
    p5_out aug2 p5_out_train || echo "[P5R] FAILED AGAIN"
  echo "[P5R] wall $(( ($(date +%s)-s)/60 )) min"
fi
D5=$(detail p5_out/eval_20000)
echo "[P5R] detail ${D5:-n/a}%   splats $(splats p5_out/site_20000.ply)"
.venv/bin/python scripts/detail_stratified.py p5_out/eval_20000 40 2>/dev/null | tail -9

# --- choose the merged config -----------------------------------------------
# Baseline to beat is P2 at 20.3. Only a clear win (>0.5) justifies changing
# the config for a 30h run; anything inside noise defaults to P2, the simplest.
BEST="--split-at-screen-size 0.03"; CAP=16000000; TAG=P2; BESTV=20.3
if [ -n "$D5" ] && awk "BEGIN{exit !($D5 > $BESTV + 0.5)}"; then
  BESTV=$D5; CAP=22000000; TAG=P5R
fi
if [ -n "$D3" ] && awk "BEGIN{exit !($D3 > $BESTV + 0.5)}"; then
  BESTV=$D3; TAG=P3
  BEST="--split-at-screen-size 0.03 --lpips-loss-weight 0.1 --lpips-crop 256"
  CAP=16000000
fi
echo "--- merged config: $TAG ($BESTV%)  cap $CAP  args: $BEST ---"

wait_gpu
echo "--- [D] MERGED 60k @ $CAP  START $(date +%T) ---"
s=$(date +%s)
EXTRA_ARGS="--growth-stop-iter 30000 $BEST" MAXRES=1920 CACHE=8GiB \
  bash scripts/run_training2.sh colmap/merged/sparse/0 60000 $CAP \
  merged_final_out merged merged_final_train
echo "[D] wall $(( ($(date +%s)-s)/3600 ))h"
.venv/bin/python scripts/sharpness.py merged_final_out/eval_60000 40 || true
.venv/bin/python scripts/detail_stratified.py merged_final_out/eval_60000 40 || true

# --- build and deploy the viewer asset --------------------------------------
UP=$(.venv/bin/python - colmap/merged/sparse/0 <<'PY'
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from colmap_io import read_points3D, read_images, qvec2R
from find_up import ground_normal
sp = sys.argv[1]
im = read_images(f'{sp}/images.bin')
C = np.array([-qvec2R(x['q']).T @ x['t'] for x in im.values()])
_, P0, _, _, _ = read_points3D(f'{sp}/points3D.bin')
n, _ = ground_normal(P0, C)
print(','.join(f'{v:.9f}' for v in n))
PY
)
.venv/bin/python scripts/check_up.py merged_final_out/site_60000.ply colmap/merged/sparse/0 derived="$UP"
.venv/bin/python scripts/ply2splat.py merged_final_out/site_60000.ply \
  -o viewer/merged_web.splat --up "$UP" --colmap colmap/merged/sparse/0 \
  --max-splats 3000000 --cameras-out viewer/merged_cameras.json
.venv/bin/python scripts/trajectory.py --colmap colmap/merged/sparse/0 \
  --cameras viewer/merged_cameras.json -o viewer/merged_traj.json
for f in merged_web.splat merged_cameras.json merged_traj.json; do
  rm -f work/webroot/$f && ln viewer/$f work/webroot/$f; done
echo "=== chain_blur4 DONE $(date +%T) -- reload 127.0.0.1:8787 ==="
