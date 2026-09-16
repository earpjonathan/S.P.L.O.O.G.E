#!/bin/bash
set -e
P=~/Desktop/fpv-splat
MODEL=$1                      # e.g. colmap/site/sparse/0
ITERS=${2:-30000}
SPLATS=${3:-6000000}
OUTDIR=${4:-site_out}
FRAMESET=${5:-site}       # frames/<set>/
DS=$P/${6:-site_train}

echo "=== building Brush dataset from $MODEL  $(date +%T) ==="
rm -rf "$DS"; mkdir -p "$DS/sparse/0" "$DS/images"
cp "$P/$MODEL"/{cameras.bin,images.bin,points3D.bin} "$DS/sparse/0/"

# hard-link the registered frames only (instant, no 1GB copy)
$P/.venv/bin/python - "$MODEL" "$FRAMESET" "$DS" <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from colmap_io import read_images
P=os.path.expanduser('~/Desktop/fpv-splat')
im=read_images(f'{P}/{sys.argv[1]}/images.bin')
fs=sys.argv[2]; ds=sys.argv[3]
n=0
for v in im.values():
    s=f'{P}/frames/{fs}/{v["name"]}'; d=f'{ds}/images/{v["name"]}'
    if os.path.exists(s):
        os.link(s,d); n+=1
print(f'  linked {n} registered images')
PY

echo "=== TRAINING $ITERS iters  $(date +%T) ==="
echo "    cache=${CACHE:-12GiB}  res=${MAXRES:-1920}  extra=${EXTRA_ARGS:-none}"
"$P/brush/target/release/brush-cli" "$DS" \
  --total-train-iters "$ITERS" \
  --max-resolution ${MAXRES:-1920} \
  --max-splats "$SPLATS" \
  --max-scene-batch-cache-size "${CACHE:-12GiB}" \
  --eval-split-every 40 --eval-every 5000 --eval-save-to-disk \
  --export-every 10000 \
  --export-path "$P/$OUTDIR/" --export-name site_{iter}.ply \
  ${EXTRA_ARGS}
echo "=== TRAINING DONE $(date +%T) ==="
ls -la "$P/$OUTDIR/"
