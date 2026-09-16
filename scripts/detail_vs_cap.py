#!/usr/bin/env python
"""Does the viewer's splat cap throw away near-field ground detail?

ply2splat ranks by opacity x projected area and keeps the top N. Ground splats
are LOW opacity and SMALL, so they rank at the bottom -- meaning the cap may be
culling precisely the splats that carry near-field ground texture. Measures
high-frequency energy in the bottom third of low-altitude renders.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

CAMS = sys.argv[1]
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])
ref = load_splat(sys.argv[2])[0]
grid, lo, hi = terrain_grid(ref)
agl = C[:, 1] - height_at(grid, lo, hi, C[:, [0, 2]])
good = (np.abs(R[:, 1, 0]) < 0.12) & (R[:, 1, 2] > -0.35) & (R[:, 1, 2] < 0.05)
low = np.where(good & (agl < np.percentile(agl[good], 20)))[0]
picks = low[np.linspace(0, len(low) - 1, 5).astype(int)]
W, H = 640, 480
for f in sys.argv[3:]:
    xyz, sc, rgba, rot = load_splat(f)
    S3 = sigma_world(sc, rot)
    hfs, accs = [], []
    for k in picks:
        img, acc, _ = render(xyz, S3, rgba, cams[k], W, H)
        g = img[2*H//3:].mean(2)
        hfs.append((np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())/2)
        accs.append(acc[2*H//3:].mean())
    print(f'  {f.split("/")[-1]:<26} splats {len(xyz):>9,}   '
          f'bottom-3rd detail {np.mean(hfs):.5f}   opacity {np.mean(accs):.3f}')
