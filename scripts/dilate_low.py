#!/usr/bin/env python
"""Does Brush's cov_blur=0.3 (absent from main.js) explain the LOW-altitude holes?

Brush's forward adds `cov_blur` to the screen-space covariance; the web viewer
does not. Splats are therefore trained under a low-pass the viewer never
applies, so every splat renders SHARPER than training assumed -- which pushes
hole pixels further into the Gaussian tails. Tested here on the low-altitude
cohort specifically, which is where the symptom lives.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

SPLAT, CAMS = sys.argv[1], sys.argv[2]
DILATES = [float(x) for x in sys.argv[3].split(',')]
W, H = 640, 480
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])
grid, lo, hi = terrain_grid(xyz)
agl = C[:, 1] - height_at(grid, lo, hi, C[:, [0, 2]])
good = (np.abs(R[:, 1, 0]) < 0.12) & (R[:, 1, 2] > -0.35) & (R[:, 1, 2] < 0.05)
low = np.where(good & (agl < np.percentile(agl[good], 20)))[0]
picks = low[np.linspace(0, len(low) - 1, 6).astype(int)]
print(f'{len(low)} low cameras; {len(picks)} sampled', flush=True)
for d in DILATES:
    ops, hs = [], []
    for k in picks:
        _, acc, _ = render(xyz, S3, rgba, cams[k], W, H, dilate=d)
        b = acc[2 * H // 3:]
        ops.append(b.mean()); hs.append(100 * np.mean(b < 0.5))
    print(f'  dilate {d:<5} LOW op {np.mean(ops):.3f}   holes {np.mean(hs):.1f}%', flush=True)
