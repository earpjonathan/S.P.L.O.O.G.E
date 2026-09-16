#!/bin/bash
# Cemetery stage. Waits for the SD copy, extracts frames (cheap, ~3 min), then
# waits for the hilltop COLMAP to finish before starting its own so the two
# mappers never compete for cores.
P=~/Desktop/fpv-splat
cd $P
while pgrep -f "cp /Volumes/SD_Card" > /dev/null; do sleep 15; done
echo "=== cemetery clips copied $(date +%T) ==="
bash scripts/extract_sf.sh aug2 0057 0058 0059 0060
N=$(ls frames/aug2 2>/dev/null | wc -l | tr -d ' ')
echo "=== aug2 extraction: $N frames (expected 3694) $(date +%T) ==="
[ "$N" -lt 3600 ] && { echo "ABORT: aug2 frame count $N short"; exit 1; }
echo "=== waiting for hilltop COLMAP to finish before starting aug2 mapper ==="
while pgrep -f "colmap (feature_extractor|sequential_matcher|matches_importer|mapper).*aug/database" > /dev/null; do sleep 60; done
while pgrep -f "run_aug.sh" > /dev/null; do sleep 60; done
echo "=== hilltop done, starting cemetery $(date +%T) ==="
bash scripts/run_session.sh aug2 aug2 "0057 0058 0059 0060"
