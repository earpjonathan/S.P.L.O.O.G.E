#!/usr/bin/env python
"""Emit a ground-truth alignment probe for the live viewer.

Picks a COLMAP-registered frame from clip 0026, and for a spread of its
observed features writes BOTH:
  uv  -- where the feature sits in the UNDISTORTED footage, normalised 0..1
  Xv  -- the same feature's 3D point in VIEWER space (the frame the .splat and
         the trajectory live in)
The page then projects Xv through the render's own view+projection matrices; if
the result lands on uv, the overlay is aligned, and no screenshot judgement is
involved.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from colmap_io import read_images, read_points3D
from trajectory import recover_transform

MODEL = 'colmap/both/sparse/0'
FX, FY, CX, CY = 725.60519, 722.77797, 960.0, 720.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.]])
D = np.array([-0.10878, 0.00815, 0.00077, 0.00037])
NAME = sys.argv[1] if len(sys.argv) > 1 else '0026_002405'

Rlev, sfac, b, _ = recover_transform('viewer/cameras.json', MODEL)
imgs = read_images(MODEL + '/images.bin')
ids, xyz, _, _, _ = read_points3D(MODEL + '/points3D.bin')
P3 = dict(zip(ids.tolist(), xyz))

im = next(v for v in imgs.values() if v['name'].startswith(NAME))
m = im['p3d'] >= 0
obs, pids = im['xys'][m], im['p3d'][m]
und = cv2.undistortPoints(obs.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
X = np.array([P3[i] for i in pids])
Xv = (X @ Rlev.T) * sfac + b

# spread the probes over the frame instead of clustering on texture
keep, grid = [], set()
for i in np.argsort(-np.linalg.norm(und - [CX, CY], axis=1)):
    g = (int(und[i, 0] // 240), int(und[i, 1] // 240))
    if g in grid or not (0 <= und[i, 0] < 1920 and 0 <= und[i, 1] < 1440):
        continue
    grid.add(g); keep.append(i)
keep = keep[:48]
out = dict(name=im['name'], t=int(NAME.split('_')[1]) / 50.0,
           uv=[[float(und[i, 0] / 1920), float(und[i, 1] / 1440)] for i in keep],
           Xv=[[float(x) for x in Xv[i]] for i in keep])
json.dump(out, open('viewer/aligncheck.json', 'w'))
print(f'{im["name"]}  t={out["t"]:.2f}  {len(keep)} probes -> viewer/aligncheck.json')
print('Rlev det', round(float(np.linalg.det(Rlev)), 6), ' sfac', round(float(sfac), 6),
      ' b', np.round(b, 4))
