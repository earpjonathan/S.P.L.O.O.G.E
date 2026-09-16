#!/bin/bash
# Blur-directed queue.
#
# Why the original merged 60k @ 8M could not have worked:
#   * every PLY ever exported holds EXACTLY 8,000,000 splats, already at 10k
#   * growth_stop_iter defaults to 15000, so 60k refines a frozen-size set
#   * split_at_screen_size defaults to 0.5 and only 0.01% of splats ever reach
#     half the frame (scripts/screen_size.py: p99 = 0.103) -- a no-op
#   * its splits are funded from `max_splats - current`, ZERO since iter 10k,
#     so lowering it without raising the cap would also have done nothing
#
# All probes run 20k iters with growth allowed the whole way, so they share one
# LR schedule and rank fairly against C0. 20k, not 30k, because five 30k runs
# is two extra days and ranking does not need full convergence.
set -e
P=~/Desktop/fpv-splat
cd $P
exec >> $P/work/chain_blur.log 2>&1
echo "=== chain_blur start $(date +%T) ==="

wait_gpu(){ while pgrep -x brush-cli > /dev/null; do sleep 60; done; }
detail(){ .venv/bin/python scripts/sharpness.py "$1" 40 2>/dev/null \
          | sed -n 's/.*BOTTOM third.*detail retained *\([0-9.]*\)%.*/\1/p'; }
splats(){ head -c 400 "$1" 2>/dev/null | sed -n 's/^element vertex //p' | head -1; }
G="--growth-stop-iter 20000"

probe(){ # tag, outdir, extra, maxres, model, frames
  local tag="$1" out="$2" extra="$3" res="${4:-1920}"
  local model="${5:-colmap/aug2/sparse/0}" frames="${6:-aug2}"
  local cap="${7:-16000000}"
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

wait_gpu; echo "--- [0] distortion-normalised post-run $(date +%T) ---"
.venv/bin/python scripts/sharpness.py aug2_distn_out/eval_30000 40 || true

echo "--- timing probe: turn the estimates into measurements $(date +%T) ---"
bash scripts/timing_probe.sh || echo "timing probe failed, continuing"

rm -f work/probe_results.txt
probe C0 c0_out    ""                                          1920 "" "" 8000000
probe P1 p1_out    ""                                          1920
probe P2 p2_out    "--split-at-screen-size 0.03"               1920
probe P4 p4_out    "--split-at-screen-size 0.03"               2688 colmap/aug2_hires/sparse/0 aug2_hires
probe P3 p3_out    "--split-at-screen-size 0.03 --lpips-loss-weight 0.1" 1920

echo "--- probe summary $(date +%T) ---"
cat work/probe_results.txt

# pick the best and spend the long merged run on it
BEST=""; BESTV=0; BESTTAG=""
while read tag val; do
  [ -z "$val" ] && continue
  if awk "BEGIN{exit !($val > $BESTV)}"; then BESTV=$val; BESTTAG=$tag; fi
done < work/probe_results.txt
case "$BESTTAG" in
  C0) echo "STOP: nothing beat the 8M control. None of capacity, splat size, the"
      echo "      loss, or resolution is the binding constraint on near-field blur."
      exit 0 ;;
  P1) BEST="" ;;
  P2) BEST="--split-at-screen-size 0.03" ;;
  P3) BEST="--split-at-screen-size 0.03 --lpips-loss-weight 0.1" ;;
  P4) BEST="--split-at-screen-size 0.03"; HIRES=1 ;;
  *)  echo "STOP: no probe produced a number."; exit 1 ;;
esac

echo "--- [D] MERGED 60k @ 16M, winner $BESTTAG ($BESTV%)  $(date +%T) ---"
wait_gpu
if [ "$HIRES" = "1" ]; then
  echo "    winner used native resolution -- extracting merged hires frames first"
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
echo "--- detail: merged 16M/60k vs merged 8M/30k (14.8%) ---"
.venv/bin/python scripts/sharpness.py merged16m_out/eval_60000 40 || true
echo "=== chain_blur DONE $(date +%T) ==="
