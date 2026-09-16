#!/usr/bin/env python
"""Accumulated alpha in the bottom third, at specific named camera poses.

Deliberately uses NO terrain model: terrain_grid fits one height field and is
invalid on a multi-site scene. Cameras are selected by clip name instead, so
this works anywhere. Distinguishes the two failure modes the gradient metric
cannot: low alpha = ray reaches the background (see-through), high alpha with
low gradient = opaque but smeared (blur).
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render

SPLAT, CAMS, CLIP = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 6
W, H = 640, 480
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
idx = [i for i, c in enumerate(cams) if c.get('img_name', '').startswith(CLIP)
       or str(c.get('id', '')).startswith(CLIP)]
if not idx:
    idx = [i for i, c in enumerate(cams) if CLIP in json.dumps(c)[:200]]
print(f'{SPLAT.split("/")[-1]}: clip {CLIP} -> {len(idx)} cameras')
picks = [idx[i] for i in np.linspace(0, len(idx) - 1, N).astype(int)]
allacc = []
for k in picks:
    img, acc, cnt = render(xyz, S3, rgba, cams[k], W, H)
    b = acc[2 * H // 3:]
    c = cnt[2 * H // 3:]
    allacc.append(b.ravel())
    print(f'  cam {k:>5}  bottom-3rd alpha mean {b.mean():.3f}  '
          f'<0.5 {100*np.mean(b<0.5):5.1f}%   <0.1 {100*np.mean(b<0.1):5.1f}%   '
          f'splats/px med {np.median(c):.0f}')
a = np.concatenate(allacc)
print(f'  POOLED alpha percentiles [5,25,50,75,95]: {np.percentile(a,[5,25,50,75,95]).round(3)}')
print(f'  POOLED  <0.1 (essentially see-through) {100*np.mean(a<0.1):.1f}%   '
      f'<0.5 {100*np.mean(a<0.5):.1f}%   >0.9 (solid) {100*np.mean(a>0.9):.1f}%')
