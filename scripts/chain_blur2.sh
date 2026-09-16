#!/bin/bash
# Resume the blur queue from P1. C0 is already done, so it is not repeated.
#
# Change from chain_blur.sh: P3 (LPIPS) is now GUARDED. Full-frame LPIPS
# measured ~29 s/iter -- about 50x the rest of the step -- and a 256px random
# crop only brought that to ~20 s/iter contended, so the cost is NOT dominated
# by VGG's pixel count the way it should be. Until that is understood, P3 gets
# a 200-iteration affordability check and is skipped rather than allowed to
# park the queue on a run that would take weeks.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_blur.log 2>&1
echo "=== chain_blur2 (resume from P1) $(date +%T) ==="

wait_gpu(){ while pgrep -x brush-cli > /dev/null; do sleep 60; done; }
detail(){ .venv/bin/python scripts/sharpness.py "$1" 40 2>/dev/null \
          | sed -n 's/.*BOTTOM third.*detail retained *\([0-9.]*\)%.*/\1/p'; }
splats(){ head -c 400 "$1" 2>/dev/null | sed -n 's/^element vertex //p' | head -1; }
G="--growth-stop-iter 20000"

probe(){ # tag outdir extra res model frames cap
  local tag="$1" out="$2" extra="$3" res="${4:-1920}"
  local model="${5:-colmap/aug2/sparse/0}" frames="${6:-aug2}" cap="${7:-16000000}"
  if [ -d "$out/eval_20000" ]; then
    echo "[$tag] already complete, skipping"
    echo "$tag $(detail $out/eval_20000)" >> work/probe_results.txt; return
  fi
  wait_gpu
  echo "--- [$tag] $extra  res=$res cap=$cap  START $(date +%T) ---"
  local s=$(date +%s)
  EXTRA_ARGS="$G $extra" MAXRES="$res" CACHE=12GiB \
    bash scripts/run_training2.sh "$model" 20000 "$cap" "$out" "$frames" "${out}_train" \
    || { echo "[$tag] FAILED"; return; }
  local mins=$(( ($(date +%s) - s) / 60 ))
  local d=$(detail $out/eval_20000)
  echo "[$tag] detail ${d:-n/a}%   splats $(splats $out/site_20000.ply)   ${mins} min"
  echo "$tag $d" >> work/probe_results.txt
}

# C0 is still running when this script starts, so wait for it before scoring:
# reading eval_20000 too early silently records an EMPTY score, and an empty
# score is skipped by the winner loop -- the control would vanish from the
# comparison without anything looking wrong.
rm -f work/probe_results.txt
wait_gpu
echo "--- [C0] finished; scoring $(date +%T) ---"
echo "C0 $(detail c0_out/eval_20000)" >> work/probe_results.txt
echo "C0 detail $(detail c0_out/eval_20000)%  splats $(splats c0_out/site_20000.ply)"

probe P1 p1_out ""                              1920
probe P2 p2_out "--split-at-screen-size 0.03"   1920
probe P4 p4_out "--split-at-screen-size 0.03"   2688 colmap/aug2_hires/sparse/0 aug2_hires

# --- P3 affordability guard -------------------------------------------------
wait_gpu; echo "--- [P3 guard] timing 200 iters of cropped LPIPS $(date +%T) ---"
s=$(date +%s)
EXTRA_ARGS="--lpips-loss-weight 0.1 --lpips-crop 256" MAXRES=1920 CACHE=4GiB \
  bash scripts/run_training2.sh colmap/aug2/sparse/0 200 2000000 \
  lpips_guard_out aug2 lpips_guard_train > /dev/null 2>&1 || true
SPI=$(awk "BEGIN{printf \"%.2f\", ($(date +%s)-$s)/200}")
echo "[P3 guard] ${SPI}s/iter uncontended -> 20k would take $(awk "BEGIN{printf \"%.1f\", $SPI*20000/3600}")h"
if awk "BEGIN{exit !($SPI < 1.0)}"; then
  probe P3 p3_out "--split-at-screen-size 0.03 --lpips-loss-weight 0.1 --lpips-crop 256" 1920
else
  echo "[P3] SKIPPED -- ${SPI}s/iter is unaffordable. LPIPS is fixed (it no longer"
  echo "     panics) but something other than VGG's pixel count dominates its cost;"
  echo "     a 42x smaller crop bought only ~1.45x. Needs the fixed-offset test."
fi

echo "--- probe summary $(date +%T) ---"; cat work/probe_results.txt

BESTV=0; BESTTAG=""
while read tag val; do
  [ -z "$val" ] && continue
  if awk "BEGIN{exit !($val > $BESTV)}"; then BESTV=$val; BESTTAG=$tag; fi
done < work/probe_results.txt
echo "winner: $BESTTAG at $BESTV%"

HIRES=0
case "$BESTTAG" in
  C0) echo "STOP: nothing beat the 8M control -- none of capacity, splat size or"
      echo "      resolution is the binding constraint on near-field blur."; exit 0 ;;
  P1) BEST="" ;;
  P2) BEST="--split-at-screen-size 0.03" ;;
  P3) BEST="--split-at-screen-size 0.03 --lpips-loss-weight 0.1 --lpips-crop 256" ;;
  P4) BEST="--split-at-screen-size 0.03"; HIRES=1 ;;
  *)  echo "STOP: no probe produced a number."; exit 1 ;;
esac

echo "--- [D] MERGED 60k @ 16M with $BESTTAG  $(date +%T) ---"
wait_gpu
if [ "$HIRES" = "1" ]; then
  bash scripts/extract_hires.sh aug "0050 0051 0052 0053 0055"
  mkdir -p frames/merged_hires
  for f in frames/aug_hires/* frames/aug2_hires/*; do
    ln -f "$f" frames/merged_hires/$(basename "$f") 2>/dev/null || true; done
  .venv/bin/python scripts/rescale_intrinsics.py colmap/merged/sparse/0 colmap/merged_hires/sparse/0 1.4
  EXTRA_ARGS="--growth-stop-iter 30000 $BEST" MAXRES=2688 CACHE=12GiB \
    bash scripts/run_training2.sh colmap/merged_hires/sparse/0 60000 16000000 \
    merged16m_out merged_hires merged16m_train
else
  EXTRA_ARGS="--growth-stop-iter 30000 $BEST" CACHE=12GiB \
    bash scripts/run_training2.sh colmap/merged/sparse/0 60000 16000000 \
    merged16m_out merged merged16m_train
fi
.venv/bin/python scripts/sharpness.py merged16m_out/eval_60000 40 || true
echo "=== chain_blur2 DONE $(date +%T) ==="
