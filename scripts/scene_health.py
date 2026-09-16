#!/usr/bin/env python
"""Scene-compaction check for a .splat.

The depth-distortion loss at too high a weight collapses the reconstruction's
depth range -- radius p99 went 17.45 -> 7.78 at w=0.05, which emptied the near
field because the cameras are fixed and the ground pulled away from them. That
signature is invisible in PSNR and in the slab metrics, so check it directly.
"""
import sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat

REF_P99 = 17.45   # baseline aug2 30k/8M
for f in sys.argv[1:]:
    xyz, sc, rgba, rot = load_splat(f)
    ok = np.isfinite(xyz).all(1)
    xyz = xyz[ok]; sc = sc[ok]
    r = np.linalg.norm(xyz - np.median(xyz, 0), axis=1)
    p50, p90, p99 = np.percentile(r, [50, 90, 99])
    ss = np.sort(sc, axis=1)
    flag = '' if p99 > 0.75 * REF_P99 else '   *** COMPACTED ***'
    print(f'{f.split("/")[-1]:<28} radius p50 {p50:5.2f}  p90 {p90:5.2f}  '
          f'p99 {p99:6.2f}  s_min med {np.median(ss[:,0]):.5f}{flag}')
