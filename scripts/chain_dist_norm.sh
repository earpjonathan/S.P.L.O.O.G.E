#!/bin/bash
# Depth-distortion retry with PER-RAY NORMALISATION.
#
# w=0.05 unnormalised collapsed the scene: radius p99 17.45 -> 7.78, LOW holes
# 33.5% -> 88.8%. Cause is that L_d is degree-1 in t and unbounded, so DISTANT
# rays have larger absolute spread and dominate the gradient -- and indeed it
# was the distant structure that got pulled in. Lowering the weight uniformly
# would weaken the term in the near-field ground too, where we actually want
# it. Normalising by each ray's own mean depth makes the term scale-free, so
# near and far rays contribute comparably. Weight kept at 0.05: under
# normalisation that is roughly the pressure NEAR rays already survived.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_dist_norm.log 2>&1
echo "=== chain_dist_norm start $(date +%T) ==="
while pgrep -x brush-cli > /dev/null; do sleep 120; done
echo "--- GPU free $(date +%T) ---"

EXTRA_ARGS="--distortion-weight 0.05 --distortion-start-iter 3000 --distortion-normalize" \
CACHE=12GiB bash scripts/run_training.sh colmap/aug2/sparse/0 30000 8000000 \
  aug2_distn_out aug2 aug2_distn_train

echo "--- measuring $(date +%T) ---"
bash scripts/eval_run.sh aug2_distn_out/site_30000.ply distn30k
.venv/bin/python scripts/scene_health.py work/eval_base8M_30k.splat work/eval_distn30k.splat
printf "distn  "; .venv/bin/python scripts/measure.py work/eval_distn30k.splat \
  viewer/aug2_cameras.json work/eval_base8M_30k.splat
printf "base   "; echo "LOW op 0.654  holes 33.5%   (reference)"
.venv/bin/python scripts/psnr_only.py aug2_distn_out/eval_30000 | head -2
echo "=== chain_dist_norm DONE $(date +%T) ==="
