#!/bin/bash
# Blur queue, revised after C0 vs P1.
#
# WHAT CHANGED. `detail retained` conflates two things: the bottom third scores
# 18% and the top half 40%, but the bottom third's GT also carries 2.44x the
# high-frequency energy (close grass vs sky and distant hills). Stratifying both
# regions by LOCAL GT TEXTURE and comparing at MATCHED difficulty
# (scripts/detail_stratified.py) shows the near field retains a flat
# 0.44-0.46x of what the far field does, in EVERY texture band. So there is a
# real near-field-specific defect -- and C0 -> P1 (8M -> 14.5M splats) left that
# ratio completely unmoved (0.45 -> 0.45). Capacity is not the lever.
#
# The flat ratio is the signature of a SCALE problem, not a content one: at
# roughly constant world-space splat density, a near-field surface gets far
# fewer splats per SCREEN pixel than a far-field one. That is exactly what
# --split-at-screen-size addresses, which makes P2 the load-bearing probe.
# P5 follows it because P2 saturated its 16M cap by iteration 10k -- so unlike
# P1, it has a mechanism actually consuming the budget.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_blur.log 2>&1
echo "=== chain_blur3 $(date +%T) ==="

wait_gpu(){ while pgrep -x brush-cli > /dev/null; do sleep 60; done; }
detail(){ .venv/bin/python scripts/sharpness.py "$1" 40 2>/dev/null \
          | sed -n 's/.*BOTTOM third.*detail retained *\([0-9.]*\)%.*/\1/p'; }
splats(){ head -c 400 "$1" 2>/dev/null | sed -n 's/^element vertex //p' | head -1; }

probe(){ # tag outdir extra res model frames cap cache
  local tag="$1" out="$2" extra="$3" res="${4:-1920}"
  local model="${5:-colmap/aug2/sparse/0}" frames="${6:-aug2}"
  local cap="${7:-16000000}" cache="${8:-12GiB}"
  if [ ! -d "$out/eval_20000" ]; then
    # wait_gpu FIRST, then re-check: a probe can be running right now, and the
    # pre-wait check would then be stale by the time the GPU frees up -- the
    # chain would wait for it to finish and immediately retrain it from scratch.
    wait_gpu
  fi
  if [ ! -d "$out/eval_20000" ]; then
    echo "--- [$tag] $extra  res=$res cap=$cap  START $(date +%T) ---"
    local s=$(date +%s)
    EXTRA_ARGS="--growth-stop-iter 20000 $extra" MAXRES="$res" CACHE="$cache" \
      bash scripts/run_training2.sh "$model" 20000 "$cap" "$out" "$frames" "${out}_train" \
      || { echo "[$tag] FAILED"; return; }
    echo "[$tag] wall $(( ($(date +%s) - s) / 60 )) min"
  else
    echo "[$tag] already complete, scoring only"
  fi
  local d=$(detail $out/eval_20000)
  echo "[$tag] detail ${d:-n/a}%   splats $(splats $out/site_20000.ply)"
  echo "[$tag] near-field deficit at matched GT texture:"
  .venv/bin/python scripts/detail_stratified.py $out/eval_20000 40 2>/dev/null | tail -9
  echo "$tag $d" >> work/probe_results.txt
}

rm -f work/probe_results.txt
probe C0 c0_out ""                            1920 "" "" 8000000
probe P1 p1_out ""                            1920
probe P2 p2_out "--split-at-screen-size 0.03" 1920
# P5: same mechanism as P2, more budget. P2 pinned 16M by iter 10k, so this is
# the capacity question asked properly -- with something consuming the budget.
probe P5 p5_out "--split-at-screen-size 0.03" 1920 "" "" 24000000 8GiB
probe P4 p4_out "--split-at-screen-size 0.03" 2688 colmap/aug2_hires/sparse/0 aug2_hires

wait_gpu; echo "--- [P3 guard] timing 200 iters of cropped LPIPS $(date +%T) ---"
s=$(date +%s)
EXTRA_ARGS="--lpips-loss-weight 0.1 --lpips-crop 256" MAXRES=1920 CACHE=4GiB \
  bash scripts/run_training2.sh colmap/aug2/sparse/0 200 2000000 \
  lpips_guard_out aug2 lpips_guard_train > /dev/null 2>&1 || true
SPI=$(awk "BEGIN{printf \"%.2f\", ($(date +%s)-$s)/200}")
echo "[P3 guard] ${SPI}s/iter uncontended -> 20k would be $(awk "BEGIN{printf \"%.1f\", $SPI*20000/3600}")h"
if awk "BEGIN{exit !($SPI < 1.0)}"; then
  probe P3 p3_out "--split-at-screen-size 0.03 --lpips-loss-weight 0.1 --lpips-crop 256" 1920
else
  echo "[P3] SKIPPED -- unaffordable at ${SPI}s/iter."
fi

echo "--- probe summary $(date +%T) ---"; cat work/probe_results.txt
BESTV=0; BESTTAG=""
while read tag val; do
  [ -z "$val" ] && continue
  if awk "BEGIN{exit !($val > $BESTV)}"; then BESTV=$val; BESTTAG=$tag; fi
done < work/probe_results.txt
echo "winner: $BESTTAG at $BESTV%"

HIRES=0; CAP=16000000
case "$BESTTAG" in
  C0|P1) echo "STOP: the winner used no near-field-specific mechanism, so there is"
         echo "      nothing worth spending 24h of merged training on. Re-read the"
         echo "      stratified tables above before queueing anything else."; exit 0 ;;
  P2) BEST="--split-at-screen-size 0.03" ;;
  P5) BEST="--split-at-screen-size 0.03"; CAP=24000000 ;;
  P3) BEST="--split-at-screen-size 0.03 --lpips-loss-weight 0.1 --lpips-crop 256" ;;
  P4) BEST="--split-at-screen-size 0.03"; HIRES=1 ;;
  *)  echo "STOP: no probe produced a number."; exit 1 ;;
esac

echo "--- [D] MERGED 60k @ $CAP with $BESTTAG  $(date +%T) ---"
wait_gpu
if [ "$HIRES" = "1" ]; then
  bash scripts/extract_hires.sh aug "0050 0051 0052 0053 0055"
  mkdir -p frames/merged_hires
  for f in frames/aug_hires/* frames/aug2_hires/*; do
    ln -f "$f" frames/merged_hires/$(basename "$f") 2>/dev/null || true; done
  .venv/bin/python scripts/rescale_intrinsics.py colmap/merged/sparse/0 colmap/merged_hires/sparse/0 1.4
  EXTRA_ARGS="--growth-stop-iter 30000 $BEST" MAXRES=2688 CACHE=8GiB \
    bash scripts/run_training2.sh colmap/merged_hires/sparse/0 60000 $CAP \
    merged16m_out merged_hires merged16m_train
else
  EXTRA_ARGS="--growth-stop-iter 30000 $BEST" CACHE=8GiB \
    bash scripts/run_training2.sh colmap/merged/sparse/0 60000 $CAP \
    merged16m_out merged merged16m_train
fi
.venv/bin/python scripts/sharpness.py merged16m_out/eval_60000 40 || true
.venv/bin/python scripts/detail_stratified.py merged16m_out/eval_60000 40 || true
echo "=== chain_blur3 DONE $(date +%T) ==="
