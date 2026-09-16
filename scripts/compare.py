#!/usr/bin/env python
"""Before/after render of the same cameras from two splat files."""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at
from PIL import Image

A, B, CAMS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
W, H = 480, 360
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])

sets = {}
for tag, path in (('before', A), ('after', B)):
    xyz, sc, rgba, rot = load_splat(path)
    sets[tag] = (xyz, sigma_world(sc, rot), rgba)

xyz = sets['before'][0]
grid, lo, hi = terrain_grid(xyz)
agl = C[:,1] - height_at(grid, lo, hi, C[:,[0,2]])
good = (np.abs(R[:,1,0]) < 0.12) & (R[:,1,2] > -0.35) & (R[:,1,2] < 0.05)
idx = np.where(good & (agl < np.percentile(agl[good], 20)))[0]
picks = idx[np.linspace(0, len(idx)-1, 3).astype(int)]

rows = []
for k in picks:
    tiles = []
    for tag in ('before','after'):
        x, S, rg = sets[tag]
        img, acc = render(x, S, rg, cams[k], W, H)
        b = acc[2*H//3:]
        print(f"  cam {k:>5} {cams[k]['img_name']} AGL {agl[k]:.3f}  {tag:<7} "
              f"bottom-3rd opacity {b.mean():.3f}  holes<0.5 {100*np.mean(b<0.5):5.1f}%")
        t = np.clip(img + (1-acc)[:,:,None]*np.array([1,0.1,0.1]), 0, 1)
        tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
    print()
grid_img = np.concatenate(rows, 0)
Image.fromarray((grid_img*255).astype(np.uint8)).save(OUT)
print("left = before, right = after ->", OUT)
