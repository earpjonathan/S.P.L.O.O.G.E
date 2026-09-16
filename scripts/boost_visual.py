#!/usr/bin/env python
"""Side-by-side render of two .splat files from the same low-altitude cameras.

Holes (accumulated alpha < 1) are tinted RED, so see-through ground is visible
at a glance rather than only in a summary statistic.

  boost_visual.py <before.splat> <after.splat> <cameras.json> <out.png>
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at
from PIL import Image

BEFORE, AFTER, CAMS, OUT = sys.argv[1:5]
W, H = 640, 480
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams])
R = np.array([c['rotation'] for c in cams])

xb, scb, rgb_, rotb = load_splat(BEFORE)
xa, sca, rga, rota = load_splat(AFTER)
Sb, Sa = sigma_world(scb, rotb), sigma_world(sca, rota)

# Same camera selection rule as measure.py: level-ish views in the lowest
# altitude quintile, which is where the ground reads as transparent.
grid, lo, hi = terrain_grid(xb)
agl = C[:, 1] - height_at(grid, lo, hi, C[:, [0, 2]])
good = (np.abs(R[:, 1, 0]) < 0.12) & (R[:, 1, 2] > -0.35) & (R[:, 1, 2] < 0.05)
low = np.where(good & (agl < np.percentile(agl[good], 20)))[0]
picks = low[np.linspace(0, len(low) - 1, 3).astype(int)]
print(f'low-altitude cameras: {len(low)}, rendering {list(picks)}', flush=True)

rows = []
for k in picks:
    tiles = []
    for tag, (x, S, rg) in (('before', (xb, Sb, rgb_)), ('after', (xa, Sa, rga))):
        img, acc, _ = render(x, S, rg, cams[k], W, H)
        b = acc[2 * H // 3:]
        print(f'  cam {k:>5} {tag:<6} bottom-3rd {b.mean():.3f}  '
              f'holes {100*np.mean(b < 0.5):.1f}%', flush=True)
        tiles.append(np.clip(img + (1 - acc)[:, :, None] * np.array([1, .1, .1]), 0, 1))
    rows.append(np.concatenate(tiles, 1))
Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(OUT)
print(f'left=before  right=after  -> {OUT}', flush=True)
