#!/usr/bin/env python
"""Copy a COLMAP model with intrinsics rescaled for higher-resolution images.

COLMAP poses are scale-invariant: reconstructing at 1920 and then training at
2688 needs no re-run, only intrinsics consistent with the new image size.
For OPENCV/RADIAL-family models the distortion coefficients act on NORMALISED
camera coordinates, so only fx, fy, cx, cy scale -- k1, k2, p1, p2 are left
alone. Getting that wrong would distort every ray while still looking sane.

  usage: rescale_intrinsics.py <src-model> <dst-model> <scale>
"""
import os, shutil, struct, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/Desktop/fpv-splat/scripts'))
from colmap_io import read_cameras

SRC, DST, S = sys.argv[1], sys.argv[2], float(sys.argv[3])
NP = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:12, 7:5, 8:4, 9:5, 10:12, 11:5}
# how many leading params are pixel-valued (f.., c..) for each model id
NPIX = {0:3, 1:4, 2:3, 3:3, 4:4, 5:4, 6:4, 7:4, 8:3, 9:3, 10:4, 11:4}

os.makedirs(DST, exist_ok=True)
cams = read_cameras(f'{SRC}/cameras.bin')

with open(f'{DST}/cameras.bin', 'wb') as f:
    f.write(struct.pack('<Q', len(cams)))
    for cid, c in sorted(cams.items()):
        m = c['model']
        w, h = int(round(c['w'] * S)), int(round(c['h'] * S))
        p = np.array(c['params'], dtype=np.float64).copy()
        npix = NPIX.get(m, 4)
        p[:npix] *= S
        assert len(p) == NP.get(m, 4), f'param count {len(p)} != {NP.get(m)} for model {m}'
        f.write(struct.pack('<iiQQ', cid, m, w, h))
        f.write(struct.pack('<' + 'd' * len(p), *p))
        print(f'  cam {cid} model {m}: {c["w"]}x{c["h"]} -> {w}x{h}'
              f'   f {c["params"][0]:.2f} -> {p[0]:.2f}'
              f'   scaled first {npix} params, left {len(p)-npix} distortion terms alone')

for n in ('images.bin', 'points3D.bin'):
    shutil.copy(f'{SRC}/{n}', f'{DST}/{n}')
print(f'  copied images.bin + points3D.bin unchanged -> {DST}')

back = read_cameras(f'{DST}/cameras.bin')
assert len(back) == len(cams)
for cid in cams:
    assert back[cid]['w'] == int(round(cams[cid]['w'] * S)), 'width round-trip failed'
    assert np.allclose(back[cid]['params'][:NPIX.get(cams[cid]['model'],4)],
                       np.array(cams[cid]['params'][:NPIX.get(cams[cid]['model'],4)]) * S)
print('  round-trip verified')
