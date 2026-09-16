#!/usr/bin/env python
"""Are the see-through regions cap-limited or observation-limited?

If raising --max-splats would help, the sparse regions should be sparse because
densification ran out of budget. If they are sparse because few CAMERAS ever saw
that ground, more budget goes wherever the gradient already is and never reaches
them. Correlates per-pose alpha against local camera coverage AND local splat
density, which separates the two.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render

SPLAT, CAMS = sys.argv[1], sys.argv[2]
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams])
W, H = 640, 480
POSES = [2500, 7448, 924, 660, 1056, 2443, 2472, 2415, 792, 396, 264, 2823]
print(f'{"cam":>6}{"alpha":>8}{"splt/px":>9}{"cams<0.4":>10}{"cams<0.8":>10}{"splats<0.3":>12}')
rows = []
for k in POSES:
    img, acc, cnt = render(xyz, S3, rgba, cams[k], W, H)
    b = acc[2*H//3:]
    P = C[k]
    dc = np.linalg.norm(C - P, axis=1)
    ncam4 = int((dc < 0.4).sum()); ncam8 = int((dc < 0.8).sum())
    ds = np.linalg.norm(xyz - P, axis=1)
    nspl = int((ds < 0.3).sum())
    rows.append((k, b.mean(), np.median(cnt[2*H//3:]), ncam4, ncam8, nspl))
    print(f'{k:>6}{b.mean():>8.3f}{rows[-1][2]:>9.0f}{ncam4:>10}{ncam8:>10}{nspl:>12,}')
a = np.array([r[1] for r in rows])
print()
for nm, j in (('splats/px', 2), ('cams within 0.4', 3), ('cams within 0.8', 4), ('splats within 0.3', 5)):
    v = np.array([r[j] for r in rows], float)
    print(f'  corr(alpha, {nm:<18}) {np.corrcoef(a, v)[0,1]:+.2f}')
