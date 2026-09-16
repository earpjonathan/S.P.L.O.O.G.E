#!/usr/bin/env python
"""Is the near-field detail gap a DEFECT, or just harder content?

`sharpness.py` divides render high-frequency energy by GT high-frequency
energy, separately for the bottom third and the top half. The bottom third
scores far worse -- but its GT also carries ~2.5x the high-frequency energy
(close grass vs sky and distant hills). Grass at 2 m is largely NOT
multi-view consistent: blades self-occlude and change with parallax, so no
amount of splats or pixels can reconstruct it.

So the ratio conflates two things. This stratifies both regions by LOCAL GT
TEXTURE and compares them at MATCHED difficulty. If retention is equal
bin-for-bin, there is no near-field defect at all and the whole blur programme
is aimed at content that cannot be reconstructed.
"""
import sys, os, glob
import numpy as np
from PIL import Image

evd = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 40

def hf(x):
    """local high-frequency energy, per pixel (mean |grad| in a 8x8 block)"""
    gx = np.abs(np.diff(x, axis=1, append=x[:, -1:]))
    gy = np.abs(np.diff(x, axis=0, append=x[-1:, :]))
    g = (gx + gy) / 2
    H, W = g.shape
    h, w = H // 8 * 8, W // 8 * 8
    return g[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3))

rows = []
for f in sorted(glob.glob(os.path.join(evd, '*.png')))[:lim]:
    name = os.path.basename(f)[:-4]
    gt = None
    for fs in sorted(glob.glob('frames/*')):
        c = os.path.join(fs, name)
        if os.path.exists(c): gt = c; break
    if gt is None: continue
    r = Image.open(f).convert('L'); g = Image.open(gt).convert('L')
    if g.size != r.size: g = g.resize(r.size, Image.LANCZOS)
    a = np.asarray(r, np.float64) / 255.0
    b = np.asarray(g, np.float64) / 255.0
    ra, rb = hf(a), hf(b)
    H = rb.shape[0]
    near = np.zeros(rb.shape, bool); near[2 * H // 3:] = True
    far  = np.zeros(rb.shape, bool); far[:H // 2] = True
    rows.append((ra, rb, near, far))

if not rows:
    print('no GT matched'); sys.exit()

RA = np.concatenate([r[0].ravel() for r in rows])
RB = np.concatenate([r[1].ravel() for r in rows])
NEAR = np.concatenate([r[2].ravel() for r in rows])
FAR = np.concatenate([r[3].ravel() for r in rows])
print(f'  n={len(rows)} images, {len(RA):,} 8x8 blocks')
print(f'  GT texture: bottom third mean {RB[NEAR].mean():.4f}   top half {RB[FAR].mean():.4f}'
      f'   ratio {RB[NEAR].mean()/RB[FAR].mean():.2f}x')

edges = np.percentile(RB[RB > 0], [0, 20, 40, 60, 80, 90, 96, 100])
print('\n  detail retained, stratified by GT texture of the same block')
print(f'  {"GT texture band":<22}{"blocks near":>12}{"near":>9}{"blocks far":>12}{"far":>9}   near/far')
for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    m = (RB >= lo) & (RB < hi)
    mn, mf = m & NEAR, m & FAR
    if mn.sum() < 500 or mf.sum() < 500: continue
    rn = RA[mn].mean() / RB[mn].mean()
    rf = RA[mf].mean() / RB[mf].mean()
    print(f'  {lo:.4f}-{hi:.4f}      {mn.sum():>10,}{100*rn:>8.1f}%{mf.sum():>12,}{100*rf:>8.1f}%'
          f'      {rn/rf:.2f}')
