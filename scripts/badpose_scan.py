#!/usr/bin/env python
"""Which poses go see-through, and what do they have in common?

No terrain model (invalid on multi-site). Correlates bottom-third alpha against
camera height relative to nearby splats, pitch, and local splat density.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render

SPLAT, CAMS = sys.argv[1], sys.argv[2]
CLIPS = sys.argv[3].split(',')
N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
W, H = 640, 480
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
blob = [json.dumps(c)[:200] for c in cams]
rows = []
for clip in CLIPS:
    idx = [i for i, b in enumerate(blob) if clip in b]
    if not idx: continue
    for k in [idx[i] for i in np.linspace(0, len(idx)-1, N).astype(int)]:
        c = cams[k]
        img, acc, cnt = render(xyz, S3, rgba, c, W, H)
        b = acc[2*H//3:]; ct = cnt[2*H//3:]
        P = np.array(c['position']); R = np.array(c['rotation'])
        # local ground height: median y of splats within 0.25 horizontally
        d = np.linalg.norm(xyz[:, [0,2]] - P[[0,2]], axis=1)
        near = xyz[d < 0.25, 1]
        gh = np.median(near) if len(near) > 50 else np.nan
        agl = gh - P[1]              # viewer y is DOWN, so ground_y - cam_y
        pitch = np.degrees(np.arcsin(np.clip(R[1,2], -1, 1)))
        rows.append((clip, k, b.mean(), 100*np.mean(b<0.5), np.median(ct), agl, pitch))
rows.sort(key=lambda r: r[2])
print(f'{"clip":<6}{"cam":>6}{"alpha":>8}{"<0.5%":>8}{"splt/px":>9}{"AGL":>8}{"pitch":>8}')
for r in rows:
    print(f'{r[0]:<6}{r[1]:>6}{r[2]:>8.3f}{r[3]:>8.1f}{r[4]:>9.0f}{r[5]:>8.3f}{r[6]:>8.1f}')
a = np.array([r[2] for r in rows]); g = np.array([r[5] for r in rows])
d = np.array([r[4] for r in rows]); p = np.array([r[6] for r in rows])
m = np.isfinite(g)
print(f'\ncorr(alpha, AGL)      {np.corrcoef(a[m],g[m])[0,1]:+.2f}')
print(f'corr(alpha, splats/px) {np.corrcoef(a,d)[0,1]:+.2f}')
print(f'corr(alpha, pitch)     {np.corrcoef(a,p)[0,1]:+.2f}')
print(f'fraction of poses with >20% bottom-third holes: {100*np.mean(np.array([r[3] for r in rows])>20):.0f}%')
