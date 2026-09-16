#!/bin/bash
set -e
P=~/Desktop/fpv-splat
OUT=$P/frames/site
rm -rf $OUT; mkdir -p $OUT
for C in 0026 0027; do
  echo "=== decoding $C $(date +%T) ==="
  TMP=$P/work/all_$C
  rm -rf $TMP; mkdir -p $TMP
  ffmpeg -v error -i $P/clips/$C.MP4 -vf "scale=1920:1440" -qscale:v 2 -start_number 0 $TMP/%06d.jpg -y
  echo "  decoded $(ls $TMP | wc -l) frames $(date +%T)"
  $P/.venv/bin/python - "$C" <<'PY'
import numpy as np, os, shutil, sys
c=sys.argv[1]; P=os.path.expanduser('~/Desktop/fpv-splat')
keep=np.load(f'{P}/work/keep_{c}.npy')
src=f'{P}/work/all_{c}'; dst=f'{P}/frames/site'
n=0
for i in keep:
    s=f'{src}/{int(i):06d}.jpg'
    if os.path.exists(s):
        shutil.move(s, f'{dst}/{c}_{int(i):06d}.jpg'); n+=1
print(f'  kept {n} / {len(keep)}')
PY
  rm -rf $TMP
done
echo "=== TOTAL $(ls $OUT | wc -l) frames, $(du -sh $OUT | cut -f1) — DONE $(date +%T) ==="
