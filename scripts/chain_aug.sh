#!/bin/bash
# Wait for frame extraction to finish, sanity-check the count, then run COLMAP.
# Detached with nohup so a 6h mapper is not tied to anything interactive.
P=~/Desktop/fpv-splat
cd $P
while pgrep -f "extract_sf.sh aug" > /dev/null; do sleep 20; done
N=$(ls frames/aug 2>/dev/null | wc -l | tr -d ' ')
echo "=== extraction finished with $N frames (expected 3777) $(date +%T) ==="
if [ "$N" -lt 3700 ]; then
  echo "ABORT: frame count $N is short of 3777 -- not starting COLMAP on a partial set."
  tail -20 reports/extract_aug.log
  exit 1
fi
bash scripts/run_aug.sh
