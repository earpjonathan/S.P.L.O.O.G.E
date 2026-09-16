#!/bin/bash
# Retry the depth-distortion loss at a much lower weight, after the merged
# training finishes. w=0.05 compressed the scene (radius p99 17.45 -> 7.78) and
# emptied the near field; the 1000-iter probe put the "free" weight near 0.01.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_dist_low.log 2>&1
echo "=== chain_dist_low start $(date +%T) ==="
while pgrep -x brush-cli > /dev/null; do sleep 120; done
echo "--- GPU free $(date +%T) ---"
EXTRA_ARGS="--distortion-weight 0.01 --distortion-start-iter 3000" CACHE=12GiB \
  bash scripts/run_training.sh colmap/aug2/sparse/0 30000 8000000 aug2_dist2_out aug2 aug2_dist2_train
echo "--- measuring $(date +%T) ---"
bash scripts/eval_run.sh aug2_dist2_out/site_30000.ply dist2_30k
.venv/bin/python scripts/measure.py work/eval_dist2_30k.splat viewer/aug2_cameras.json work/eval_base8M_30k.splat
.venv/bin/python scripts/eval_compare.py aug2_dist2_out/eval_30000 /tmp/d2.jpg | grep "PSNR mean"
echo "=== chain_dist_low DONE $(date +%T) ==="
