#!/usr/bin/env python
"""Compare rendered COVERAGE across splat assets at the same low-flight poses.

The detail metric (mean |grad|) measures TEXTURE and says nothing about whether
a ray accumulates alpha. `--split-at-screen-size` shrinks splats, and the viewer
then keeps only the top N by opacity x projected area -- so a run with twice the
splats has a far larger fraction DISCARDED at export, each survivor covering
less. Both effects cut coverage while detail goes up.

Renders identical poses from each asset and reports bottom-third opacity and
hole fraction, at the shipped znear=0.01.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from raster import load_splat, sigma_world, render

CAMS = sys.argv[1]
ASSETS = sys.argv[2:]
W, H, N = 640, 480, 8

cams = json.load(open(CAMS))
# lowest-lying, level-looking poses: where coverage failures show up
C = np.array([c['position'] for c in cams])
Rz = np.array([c['rotation'] for c in cams])[:, :, 2]
level = np.abs(Rz[:, 1]) < 0.45
lo = np.percentile(C[:, 1], 25)
cand = np.where((C[:, 1] < lo) & level)[0]
picks = list(cand[np.linspace(0, len(cand) - 1, N).astype(int)])
print(f'{len(cams):,} cameras, {len(cand)} low+level, rendering {N}\n')

res = {}
for path in ASSETS:
    xyz, sc, rgba, rot = load_splat(path)
    S3 = sigma_world(sc, rot)
    ops, holes = [], []
    for i in picks:
        _, acc, _ = render(xyz, S3, rgba, cams[i], W, H, znear=0.01, fade=True)
        b = acc[int(H * 2 / 3):]
        ops.append(float(b.mean())); holes.append(float(100 * np.mean(b < 0.5)))
    res[path] = (np.array(ops), np.array(holes), len(xyz))
    o, h, n = res[path]
    print(f'{os.path.basename(path):<28} {n:>10,} splats   '
          f'bottom-third opacity {o.mean():.3f}   holes {h.mean():5.1f}%   worst {h.max():5.1f}%')

if len(ASSETS) > 1:
    print('\n  per-pose holes')
    print('  pose   ' + '  '.join(f'{os.path.basename(a)[:16]:>16}' for a in ASSETS))
    for j, i in enumerate(picks):
        print(f'  {i:>6} ' + '  '.join(f'{res[a][1][j]:>15.1f}%' for a in ASSETS))
