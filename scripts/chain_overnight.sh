#!/bin/bash
# Overnight queue, in priority order.
#
# 1. Distortion retry WITH per-ray normalisation (2.8h). Answers the open
#    research question: w=0.05 unnormalised collapsed the scene because L_d is
#    degree-1 in t, so far rays dominated.
# 2. Merged at 60k (9.6h). The real payoff. Near-field ground detail is still
#    ACCELERATING at 30k -- +3.3 pts over 10k->20k, +4.4 over 20k->30k -- so
#    every run so far has stopped exactly where detail was improving fastest.
#    60k sets the LR schedule to 60k, so it is a genuinely different run, not
#    just more of the same.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_overnight.log 2>&1
echo "=== chain_overnight start $(date +%T) ==="

wait_gpu(){ while pgrep -x brush-cli > /dev/null; do sleep 120; done; }

wait_gpu; echo "--- [1/2] distortion normalised $(date +%T) ---"
EXTRA_ARGS="--distortion-weight 0.05 --distortion-start-iter 3000 --distortion-normalize" \
CACHE=12GiB bash scripts/run_training.sh colmap/aug2/sparse/0 30000 8000000 \
  aug2_distn_out aug2 aug2_distn_train || echo "distn run FAILED, continuing"
bash scripts/eval_run.sh aug2_distn_out/site_30000.ply distn30k || true
.venv/bin/python scripts/scene_health.py work/eval_base8M_30k.splat work/eval_distn30k.splat || true
printf "distn  "; .venv/bin/python scripts/measure.py work/eval_distn30k.splat \
  viewer/aug2_cameras.json work/eval_base8M_30k.splat || true
.venv/bin/python scripts/sharpness.py aug2_distn_out/eval_30000 40 || true

wait_gpu; echo "--- [2/2] MERGED 60k $(date +%T) ---"
CACHE=12GiB bash scripts/run_training.sh colmap/merged/sparse/0 60000 8000000 \
  merged60k_out merged merged60k_train
echo "--- detail at 60k vs the 30k run ---"
.venv/bin/python scripts/sharpness.py merged60k_out/eval_60000 40 || true
.venv/bin/python scripts/sharpness.py merged_out/eval_30000 40 || true
echo "=== chain_overnight DONE $(date +%T) ==="
