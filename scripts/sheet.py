#!/usr/bin/env python
"""Contact sheet of level, forward-looking cameras at a chosen altitude band."""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at
from PIL import Image

SPLAT, CAMS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 480, 360
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams])
R = np.array([c['rotation'] for c in cams])
grid, lo, hi = terrain_grid(xyz)
agl = C[:,1] - height_at(grid, lo, hi, C[:,[0,2]])

roll  = np.abs(R[:,1,0])          # world-Y component of camera RIGHT axis
pitch = R[:,1,2]                  # world-Y component of camera FORWARD axis
good = (roll < 0.12) & (pitch > -0.35) & (pitch < 0.05)
print(f"level+forward cameras: {good.sum()} / {len(cams)}")
print(f"AGL over those: median {np.median(agl[good]):.3f} "
      f"p10 {np.percentile(agl[good],10):.3f} p90 {np.percentile(agl[good],90):.3f}")

sel = []
for name, m in [("low",  good & (agl < np.percentile(agl[good],15))),
                ("mid",  good & (agl > np.percentile(agl[good],40))
                              & (agl < np.percentile(agl[good],60))),
                ("high", good & (agl > np.percentile(agl[good],85)))]:
    idx = np.where(m)[0]
    for k in idx[np.linspace(0, len(idx)-1, 2).astype(int)]:
        sel.append((name, k))

tiles = []
for name, k in sel:
    img, acc = render(xyz, S3, rgba, cams[k], W, H)
    b = acc[2*H//3:]
    print(f"  {name:>4} cam {k:>5} {cams[k]['img_name']} AGL {agl[k]:.3f} "
          f"bottom-3rd opacity {b.mean():.3f} holes<0.5 {100*np.mean(b<0.5):5.1f}%")
    t = img + (1-acc)[:,:,None]*np.array([1,0.1,0.1])   # red shows see-through
    t[2*H//3-1:2*H//3+1] = [0,0.6,1]
    tiles.append(np.clip(t,0,1))
rows = [np.concatenate(tiles[i:i+2], 1) for i in range(0, len(tiles), 2)]
Image.fromarray((np.concatenate(rows,0)*255).astype(np.uint8)).save(OUT)
print("->", OUT)
