#!/usr/bin/env python
"""Audit the viewer's znear fix on one scene.

Renders the SAME splats/cameras twice -- znear 0.2 (the old viewer) and
znear 0.01 (the fix) -- over two cohorts:

  LOW   the lowest-AGL level-looking cameras, where a fixed near plane
        culls the ground directly beneath the drone
  GOOD  cameras at median AGL, as a regression guard

Reports the cohort mean AND the worst single pose, because the defect lives
in the tail: a cohort mean is exactly what hid this bug for two days.

Only valid on single-site scenes -- terrain_grid fits ONE height field, so on
a merged multi-site scene it interpolates across empty space and every AGL
is meaningless.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

SPLAT, CAMS, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
NLOW  = int(sys.argv[4]) if len(sys.argv) > 4 else 8
NGOOD = int(sys.argv[5]) if len(sys.argv) > 5 else 5
W, H = 640, 480

xyz, sc, rgba, rot = load_splat(SPLAT)
cams = json.load(open(CAMS))
print(f"{TAG}: {len(xyz):,} splats  {len(cams):,} cameras", flush=True)
S3 = sigma_world(sc, rot)

grid, lo, hi = terrain_grid(xyz)
C   = np.array([c['position'] for c in cams])
agl = C[:, 1] - height_at(grid, lo, hi, C[:, [0, 2]])
Rz  = np.array([c['rotation'] for c in cams])[:, :, 2]
level = np.abs(Rz[:, 1]) < 0.45

if np.median(agl) <= 0 or np.mean(agl < 0) > 0.25:
    print(f"  !! AGL invalid (median {np.median(agl):.3f}, "
          f"{100*np.mean(agl<0):.0f}% underground) -- multi-site scene? aborting")
    sys.exit(1)
print(f"  median AGL {np.median(agl):.3f} scene units", flush=True)

def cohort(mask, n, name):
    idx = np.where(mask & level)[0]
    if len(idx) == 0: return []
    return list(idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)])

low  = cohort(agl < np.percentile(agl, 20), NLOW, "LOW")
mid  = np.percentile(agl, [45, 55])
good = cohort((agl > mid[0]) & (agl < mid[1]), NGOOD, "GOOD")
print(f"  LOW n={len(low)}  GOOD n={len(good)}\n", flush=True)

rows = {}
for label, picks in [("LOW", low), ("GOOD", good)]:
    res = {"old": [], "fix": []}
    for i in picks:
        cam = cams[i]
        line = f"  {label} cam {i:>5} {cam['img_name']:>16} AGL {agl[i]:.3f} "
        for key, zn in [("old", 0.2), ("fix", 0.01)]:
            t = time.time()
            _, acc, _ = render(xyz, S3, rgba, cam, W, H, znear=zn, fade=True)
            b = acc[int(H * 2 / 3):]
            op, holes = float(b.mean()), float(100 * np.mean(b < 0.5))
            res[key].append((op, holes))
            line += f"| znear {zn:<4} op {op:.3f} holes {holes:5.1f}% "
        print(line + f"[{time.time()-t:.0f}s]", flush=True)
    rows[label] = res
    print(flush=True)

print("=" * 74)
print(f"{TAG}")
for label in ("LOW", "GOOD"):
    r = rows[label]
    if not r["old"]: continue
    o, f = np.array(r["old"]), np.array(r["fix"])
    print(f"  {label:<5} n={len(o)}")
    print(f"    mean  opacity {o[:,0].mean():.3f} -> {f[:,0].mean():.3f}"
          f"    holes {o[:,1].mean():5.1f}% -> {f[:,1].mean():5.1f}%")
    print(f"    worst opacity {o[:,0].min():.3f} -> {f[:,0].min():.3f}"
          f"    holes {o[:,1].max():5.1f}% -> {f[:,1].max():5.1f}%")
    worse = int(np.sum(f[:,1] > o[:,1] + 0.5))
    print(f"    poses made worse by the fix: {worse}/{len(o)}")
