#!/bin/bash
# Measure the per-iteration cost of each candidate config, so the queue can be
# scheduled from facts instead of multipliers I made up.
#
# Startup (dataset scan, cache fill) is a fixed cost that would swamp a short
# run, so each config is timed TWICE -- at N1 and N2 iterations -- and the
# slope (t2-t1)/(N2-N1) is reported. Splat count is still climbing this early,
# so these are LOWER bounds on steady-state cost; they are exact for the
# additive costs (LPIPS, resolution) that do not depend on splat count.
set -e
P=~/Desktop/fpv-splat
cd $P
N1=600; N2=2000
run(){ # name, extra args, maxres
  local name="$1" extra="$2" res="$3" t1 t2 slope
  for n in $N1 $N2; do
    local s=$(date +%s)
    EXTRA_ARGS="$extra" MAXRES="$res" CACHE=8GiB \
      bash scripts/run_training2.sh "${MODEL:-colmap/aug2/sparse/0}" $n 16000000 \
      tp_out "${FRAMES:-aug2}" tp_train > /dev/null 2>&1 || { echo "  $name: FAILED"; return; }
    local e=$(( $(date +%s) - s ))
    if [ "$n" = "$N1" ]; then t1=$e; else t2=$e; fi
  done
  slope=$(awk "BEGIN{printf \"%.4f\", ($t2-$t1)/($N2-$N1)}")
  local h30=$(awk "BEGIN{printf \"%.1f\", $slope*30000/3600}")
  printf "  %-26s %5ss @%s  %5ss @%s   %ss/iter   -> 30k ~ %sh (lower bound)\n" \
         "$name" "$t1" "$N1" "$t2" "$N2" "$slope" "$h30"
}
echo "=== timing probe $(date +%T) ==="
run "baseline 1920"          ""                      1920
run "+ split 0.03"           "--split-at-screen-size 0.03" 1920
run "+ lpips 0.1"            "--lpips-loss-weight 0.1"     1920
MODEL=colmap/aug2_hires/sparse/0 FRAMES=aug2_hires run "native 2688"  ""  2688
rm -rf $P/tp_out $P/tp_train
echo "=== timing probe DONE $(date +%T) ==="
