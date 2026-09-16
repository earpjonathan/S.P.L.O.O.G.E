#!/usr/bin/env python
"""A/B the viewer's near-field handling on real low-flight cameras."""
import json, sys, time
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

SPLAT, CAMS, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 640, 480

xyz, sc, rgba, rot = load_splat(SPLAT)
cams = json.load(open(CAMS))
print(f"{TAG}: {len(xyz):,} splats  {len(cams):,} cameras")
t0 = time.time(); S3 = sigma_world(sc, rot); print(f"  cov {time.time()-t0:.1f}s")

grid, lo, hi = terrain_grid(xyz)
C = np.array([c['position'] for c in cams])
agl = C[:, 1] - height_at(grid, lo, hi, C[:, [0, 2]])

# pick low cameras spread through the flight, looking near-level
Rz = np.array([c['rotation'] for c in cams])[:, :, 2]     # forward axis
level = np.abs(Rz[:, 1]) < 0.45
cand = np.where((agl < np.percentile(agl, 20)) & level)[0]
picks = cand[np.linspace(0, len(cand)-1, 3).astype(int)]
print(f"  {len(cand)} low+level cameras; rendering {list(picks)}\n")

out = {}
for idx in picks:
    cam = cams[idx]
    row = {}
    for name, kw in [("current", dict(znear=0.2, fade=True)),
                     ("znear 0.01", dict(znear=0.01, fade=True)),
                     ("no fade", dict(znear=0.2, fade=False))]:
        t = time.time()
        img, acc, _cnt = render(xyz, S3, rgba, cam, W, H, **kw)
        b = acc[int(H*2/3):]          # bottom third
        m = acc[int(H/3):int(H*2/3)]  # middle third
        row[name] = (img, acc)
        print(f"  cam {idx} {cam['img_name']:>14} AGL {agl[idx]:.3f}  "
              f"{name:<11} bottom-3rd opacity {b.mean():.3f}  "
              f"holes(<0.5) {100*np.mean(b<0.5):5.1f}%  "
              f"mid {m.mean():.3f}  [{time.time()-t:.0f}s]")
    out[idx] = row
    print()

np.save(f'/Users/jonathanearp/Desktop/fpv-splat/work/ab_{TAG}.npy',
        {k: {n: v for n, v in r.items()} for k, r in out.items()},
        allow_pickle=True)
