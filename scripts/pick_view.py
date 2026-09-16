#!/usr/bin/env python
"""Pick the opening camera for the viewer: the capture pose that frames the most
of the reconstructed scene. Reads the exported .splat (already in viewer frame)
and the cameras.json emitted alongside it."""
import sys, json, numpy as np

splat, camj = sys.argv[1], sys.argv[2]
d = np.fromfile(splat, dtype=np.uint8).reshape(-1, 32)
pos = d[:, :12].copy().view('<f4').reshape(-1, 3)
op  = d[:, 24 + 3].astype(np.float32) / 255.0
rng = np.random.default_rng(0)
sel = rng.choice(len(pos), min(200_000, len(pos)), replace=False)
P, O = pos[sel], op[sel]

cams = json.load(open(camj))
# ply2splat levels the scene so the vertical axis maps to (0,-1,0)
scores = []
for i, c in enumerate(cams):
    C = np.array(c['position'], float)
    R = np.array(c['rotation'], float)          # camera-to-world, row-major
    fwd = R[:, 2]
    v = P - C
    z = v @ fwd
    front = z > 0
    if front.sum() < 1000:
        scores.append((i, 0.0, 0.0, 0.0)); continue
    fx, fy = c['fx'], c['fy']; W, H = c['width'], c['height']
    x = (v @ R[:, 0]) / np.maximum(z, 1e-6) * fx
    y = (v @ R[:, 1]) / np.maximum(z, 1e-6) * fy
    inside = front & (np.abs(x) < W / 2) & (np.abs(y) < H / 2)
    frac = inside.mean()
    # prefer views that are not mostly sky: weight by opacity mass in frame
    mass = float(O[inside].sum()) / max(O.sum(), 1e-6)
    # and prefer looking down-ish (levelled scene: world up is -Y)
    down = float(-fwd[1])
    scores.append((i, frac, mass, down))

scores.sort(key=lambda s: -(s[1] * 0.5 + s[2] * 0.5))
print(f'{len(cams)} cameras scored (frac = splats in frustum, mass = opacity share)')
print(f'{"idx":>4} {"frame":>14} {"frac":>7} {"mass":>7} {"down":>6}')
for i, frac, mass, down in scores[:10]:
    print(f'{i:>4} {cams[i]["img_name"]:>14} {frac:7.3f} {mass:7.3f} {down:6.2f}')
print(f'\nrecommended: ?cam={scores[0][0]}  ({cams[scores[0][0]]["img_name"]})')
