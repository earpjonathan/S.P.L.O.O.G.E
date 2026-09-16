#!/usr/bin/env python
"""Distribution of splat SCREEN SIZE, in the same units --split-at-screen-size uses.

Brush's `max_screen_size` is the larger 2D ellipse extent as a FRACTION OF THE
IMAGE DIMENSION, tracked as a running max over every view a splat is visible in.
`--split-at-screen-size` splits anything above the threshold and shrinks the
children to land at (at most) it. Default 0.5 -- half the frame.

This reproduces that statistic from an exported PLY plus the COLMAP cameras, so
a threshold can be chosen from the actual distribution instead of guessed.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from ply2splat import read_ply
from colmap_io import read_images, read_cameras, qvec2R

PLY, MODEL = sys.argv[1], sys.argv[2]
NSPLAT = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
NCAM   = int(sys.argv[4]) if len(sys.argv) > 4 else 150

d, n, _ = read_ply(PLY)
print(f'{PLY}: {n:,} splats', flush=True)
rng = np.random.default_rng(0)
sub = rng.choice(n, size=min(NSPLAT, n), replace=False)
xyz = np.stack([d['x'][sub], d['y'][sub], d['z'][sub]], 1).astype(np.float64)
# scales are stored log-encoded; the largest axis bounds the projected extent
sc  = np.exp(np.stack([d['scale_0'][sub], d['scale_1'][sub], d['scale_2'][sub]], 1)).astype(np.float64)
smax = sc.max(1)
opac = 1.0 / (1.0 + np.exp(-d['opacity'][sub].astype(np.float64)))

ims  = read_images(f'{MODEL}/images.bin')
cams = read_cameras(f'{MODEL}/cameras.bin')
keys = sorted(ims.keys())
pick = keys[:: max(1, len(keys) // NCAM)][:NCAM]
print(f'  projecting into {len(pick)} of {len(keys):,} cameras', flush=True)

run_max = np.zeros(len(xyz))
seen    = np.zeros(len(xyz), bool)
for k in pick:
    im = ims[k]
    cam = cams[im['cid']]
    fx, fy = cam['params'][0], cam['params'][1]
    W, H = cam['w'], cam['h']
    R = qvec2R(im['q'])
    C = -R.T @ im['t']
    Xc = (xyz - C) @ R.T.T          # world -> camera
    z = Xc[:, 2]
    ok = z > 1e-3
    if not ok.any():
        continue
    # 2 sigma of the largest axis, in pixels, as a fraction of the image dim
    ext_px = 2.0 * fx * smax[ok] / z[ok]
    frac = ext_px / max(W, H)
    u = fx * Xc[ok, 0] / z[ok] + W / 2.0
    v = fy * Xc[ok, 1] / z[ok] + H / 2.0
    inview = (u > -W) & (u < 2 * W) & (v > -H) & (v < 2 * H)
    idx = np.where(ok)[0][inview]
    run_max[idx] = np.maximum(run_max[idx], frac[inview])
    seen[idx] = True

s = run_max[seen]
print(f'  {seen.sum():,} of {len(xyz):,} sampled splats were visible somewhere\n')
print('  max screen extent, as a fraction of the image dimension')
for p in (50, 75, 90, 95, 99, 99.9):
    print(f'    p{p:<5} {np.percentile(s, p):.4f}')
print(f'    max    {s.max():.4f}')
print('\n  fraction ABOVE each candidate --split-at-screen-size')
for t in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
    frac = float(np.mean(s > t))
    # each oversized splat costs one extra splat at the next refine
    print(f'    > {t:<5} {100*frac:6.2f}%   ~{frac*n/1e6:5.2f}M splats would split'
          + ('   <-- current default' if t == 0.5 else ''))
big = s > 0.1
if big.any():
    print(f'\n  splats above 0.1: median opacity {np.median(opac[seen][big]):.3f} '
          f'vs {np.median(opac[seen][~big]):.3f} for the rest')
