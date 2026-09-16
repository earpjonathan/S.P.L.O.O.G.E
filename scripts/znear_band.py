#!/usr/bin/env python
"""Rank cameras by the znear cull MECHANISM, then A/B render the worst.

No terrain model, so this is valid on merged multi-site scenes where
terrain_grid silently interpolates across empty space.

The viewer rejects a splat when  pz < -margin*z  with
    pz = z*zfar/(zfar-znear) - zfar*znear/(zfar-znear)
which is a pure near-depth cut at
    z < znear*k / (margin + k),   k = zfar/(zfar-znear)
For zfar=200, margin=1.2 that is z < 0.0910 at znear=0.2 and z < 0.0045 at
znear=0.01.  Splats inside that band are exactly the ones the fix recovers,
so ranking cameras by their in-frustum share of banded splats finds the poses
where znear can possibly matter -- rather than averaging it away.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render

SPLAT, CAMS, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
TOPK   = int(sys.argv[4]) if len(sys.argv) > 4 else 6
CTRL   = int(sys.argv[5]) if len(sys.argv) > 5 else 4
W, H, ZFAR, MARGIN = 640, 480, 200.0, 1.2

def zcut(znear):
    k = ZFAR / (ZFAR - znear)
    return znear * k / (MARGIN + k)

CUT_OLD, CUT_FIX = zcut(0.2), zcut(0.01)

xyz, sc, rgba, rot = load_splat(SPLAT)
cams = json.load(open(CAMS))
print(f"{TAG}: {len(xyz):,} splats  {len(cams):,} cameras")
print(f"  cull band: z in ({CUT_FIX:.4f}, {CUT_OLD:.4f})", flush=True)
S3 = sigma_world(sc, rot)

# scan on a splat subsample weighted to opaque ones -- the transparent tail
# cannot open a hole whether it is culled or not
rng = np.random.default_rng(0)
w = rgba[:, 3].astype(np.float64); w = w / w.sum()
sub = rng.choice(len(xyz), size=min(250_000, len(xyz)), replace=False, p=w)
P = xyz[sub]

stride = max(1, len(cams) // 500)
scan = list(range(0, len(cams), stride))
t0 = time.time()
frac = np.zeros(len(scan))
for j, ci in enumerate(scan):
    c = cams[ci]
    Xc = (P - np.asarray(c['position'], float)) @ np.asarray(c['rotation'], float)
    z = Xc[:, 2]
    s = H / c['height']
    px = (2 * c['fx'] * s / W) * Xc[:, 0]
    py = -(2 * c['fy'] * s / H) * Xc[:, 1]
    clip = MARGIN * z
    inf = (z > 0) & (np.abs(px) <= clip) & (np.abs(py) <= clip)
    n = inf.sum()
    frac[j] = 0.0 if n == 0 else float(((z > CUT_FIX) & (z < CUT_OLD) & inf).sum()) / n
print(f"  scanned {len(scan)} cameras in {time.time()-t0:.0f}s;"
      f" banded share p50 {np.median(frac):.4f} p99 {np.percentile(frac,99):.4f}"
      f" max {frac.max():.4f}", flush=True)

order = np.argsort(-frac)
picks  = [(scan[j], frac[j], "WORST") for j in order[:TOPK]]
midj   = order[len(order)//2 : len(order)//2 + CTRL]
picks += [(scan[j], frac[j], "CTRL") for j in midj]

res = {"WORST": {"old": [], "fix": []}, "CTRL": {"old": [], "fix": []}}
print(flush=True)
for ci, fr, kind in picks:
    cam = cams[ci]
    line = f"  {kind:<5} cam {ci:>5} {cam['img_name']:>16} banded {100*fr:5.2f}% "
    for key, zn in [("old", 0.2), ("fix", 0.01)]:
        _, acc, _ = render(xyz, S3, rgba, cam, W, H, znear=zn, fade=True)
        b = acc[int(H * 2 / 3):]
        op, holes = float(b.mean()), float(100 * np.mean(b < 0.5))
        res[kind][key].append((op, holes))
        line += f"| {zn:<4} op {op:.3f} holes {holes:5.1f}% "
    print(line, flush=True)

print("\n" + "=" * 74)
print(TAG)
for kind in ("WORST", "CTRL"):
    o, f = np.array(res[kind]["old"]), np.array(res[kind]["fix"])
    if not len(o): continue
    print(f"  {kind:<5} n={len(o)}  bottom-third holes "
          f"mean {o[:,1].mean():5.1f}% -> {f[:,1].mean():5.1f}%   "
          f"worst {o[:,1].max():5.1f}% -> {f[:,1].max():5.1f}%   "
          f"regressed {int(np.sum(f[:,1] > o[:,1] + 0.5))}/{len(o)}")
