#!/bin/bash
# Re-extract the EXACT same frames at native sensor resolution.
#
# Frames are named <clip>_<video frame index>.jpg, so selecting the same
# indices guarantees the same views -- the COLMAP poses stay valid and only the
# pixel count changes. Source is 2688x2016; the existing set is 1920x1440,
# so ~half the pixels have been thrown away before training ever saw them.
set -e
P=~/Desktop/fpv-splat
SET=${1:-aug2}
CLIPS=${2:-"0057 0058 0059 0060"}
OUT=$P/frames/${SET}_hires
mkdir -p $OUT
echo "=== extract_hires $SET -> $OUT  $(date +%T) ==="
for C in $CLIPS; do
  want=$(ls $P/frames/$SET/ | grep "^${C}_" | wc -l | tr -d ' ')
  have=$(ls $OUT 2>/dev/null | grep -c "^${C}_" || true)
  if [ "$have" = "$want" ]; then echo "  $C: already have $have, skipping"; continue; fi
  echo "  --- $C: need $want frames  $(date +%T) ---"
  TMP=$P/work/hires_$C
  rm -rf $TMP; mkdir -p $TMP
  # native resolution, no scale filter at all
  ffmpeg -v error -i $P/clips/$C.MP4 -qscale:v 2 -start_number 0 $TMP/%06d.jpg -y
  echo "      decoded $(ls $TMP | wc -l) frames  $(date +%T)"
  $P/.venv/bin/python - "$C" "$SET" <<'PY'
import os, shutil, sys, glob
c, st = sys.argv[1], sys.argv[2]
P = os.path.expanduser('~/Desktop/fpv-splat')
src, dst = f'{P}/work/hires_{c}', f'{P}/frames/{st}_hires'
names = [os.path.basename(f) for f in glob.glob(f'{P}/frames/{st}/{c}_*.jpg')]
n = 0
for nm in names:
    idx = nm.split('_')[1].split('.')[0]
    s = f'{src}/{idx}.jpg'
    if os.path.exists(s):
        shutil.move(s, f'{dst}/{nm}'); n += 1
print(f'      kept {n} / {len(names)}')
PY
  rm -rf $TMP
done
echo "=== TOTAL $(ls $OUT | wc -l) frames, $(du -sh $OUT | cut -f1)  DONE $(date +%T) ==="
