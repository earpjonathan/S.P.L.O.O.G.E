#!/bin/bash
# After the hilltop mapper exits: pick the largest sparse model, report the
# registration rate, then train. Brush is on Metal and COLMAP is on CPU, so
# this runs happily alongside the cemetery mapper chained in chain_aug2.sh.
P=~/Desktop/fpv-splat
cd $P
while pgrep -f "run_aug.sh" > /dev/null; do sleep 60; done
echo "=== hilltop mapper finished $(date +%T) ==="
BEST=$($P/.venv/bin/python - <<'PY'
import sys, os, glob
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from colmap_io import read_images
P = os.path.expanduser('~/Desktop/fpv-splat')
best, bn = None, -1
for d in sorted(glob.glob(f'{P}/colmap/aug/sparse/*')):
    try: n = len(read_images(f'{d}/images.bin'))
    except Exception: continue
    print(f'  model {os.path.basename(d)}: {n} images', file=sys.stderr)
    if n > bn: best, bn = d, n
total = len(glob.glob(f'{P}/frames/aug/*.jpg'))
print(f'  BEST {best}: {bn}/{total} = {100*bn/total:.1f}% registered', file=sys.stderr)
print(os.path.relpath(best, P) if best else '')
PY
)
echo "chosen model: $BEST"
[ -z "$BEST" ] && { echo "ABORT: no sparse model produced"; exit 1; }
bash scripts/run_training.sh "$BEST" 30000 8000000 aug_out aug aug_train
