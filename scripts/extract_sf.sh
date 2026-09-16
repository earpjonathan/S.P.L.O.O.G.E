#!/bin/bash
# Decode selected frames for a set of clips into frames/<set>/.
# Decodes each clip fully to a temp dir then keeps only the gyro-selected
# frames, deleting the temp before the next clip so peak disk stays ~1 clip.
set -e
P=~/Desktop/fpv-splat
SET=$1; shift
OUT=$P/frames/$SET
mkdir -p $OUT
for C in "$@"; do
  [ -n "$(ls $OUT/${C}_* 2>/dev/null | head -1)" ] && { echo "  $C already extracted"; continue; }
  echo "=== decoding $C $(date +%T) ==="
  TMP=$P/work/all_$C; rm -rf $TMP; mkdir -p $TMP
  ffmpeg -v error -i $P/clips/$C.MP4 -vf "scale=1920:1440" -qscale:v 2 -start_number 0 $TMP/%06d.jpg -y
  echo "  decoded $(ls $TMP | wc -l) frames $(date +%T)"
  $P/.venv/bin/python - "$C" "$SET" <<'PY'
import numpy as np, os, shutil, sys
c,s=sys.argv[1],sys.argv[2]; P=os.path.expanduser('~/Desktop/fpv-splat')
keep=np.load(f'{P}/work/keep_{c}.npy')
src=f'{P}/work/all_{c}'; dst=f'{P}/frames/{s}'
n=0
for i in keep:
    p=f'{src}/{int(i):06d}.jpg'
    if os.path.exists(p):
        shutil.move(p, f'{dst}/{c}_{int(i):06d}.jpg'); n+=1
print(f'  kept {n} / {len(keep)}')
PY
  rm -rf $TMP
done
echo "=== $SET TOTAL $(ls $OUT | wc -l) frames, $(du -sh $OUT | cut -f1) $(date +%T) ==="
