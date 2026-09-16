#!/usr/bin/env python
"""PSNR for a Brush eval dir, no montage.

eval_compare.py builds a side-by-side canvas, which overflows JPEG's 65535px
dimension limit once an eval set gets large (the merged scene has 187 views).
The scores are fine; only the save fails. This computes the numbers alone.
"""
import sys, os, glob
import numpy as np
from PIL import Image

evd = sys.argv[1]
scores = []
for f in sorted(glob.glob(os.path.join(evd, '*.png'))):
    name = os.path.basename(f)[:-4]
    gt = None
    for fs in sorted(glob.glob('frames/*')):
        c = os.path.join(fs, name)
        if os.path.exists(c): gt = c; break
    if gt is None: continue
    r = Image.open(f).convert('RGB'); g = Image.open(gt).convert('RGB')
    if g.size != r.size: g = g.resize(r.size, Image.LANCZOS)
    mse = float(((np.asarray(r, np.float64)/255 - np.asarray(g, np.float64)/255)**2).mean())
    scores.append((name, 10*np.log10(1/mse) if mse > 0 else 99.0))
ps = [s for _, s in scores]
print(f'images {len(ps)}')
if ps:
    print(f'PSNR mean {np.mean(ps):.2f} dB   median {np.median(ps):.2f}   '
          f'min {np.min(ps):.2f}   max {np.max(ps):.2f}')
    # per-clip, so a merge can be checked for one session dragging the other down
    byclip = {}
    for n, s in scores: byclip.setdefault(n[:4], []).append(s)
    for c in sorted(byclip):
        v = byclip[c]
        print(f'  {c}: n={len(v):<4} mean {np.mean(v):.2f} dB')
