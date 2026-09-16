#!/usr/bin/env python
"""Judge a cross-session merge by COLMAP's OWN two-view geometry verification.

The inlier count from a raw SIFT probe cannot tell a usable link from a useless
one. COLMAP's `config` field can: it records what model actually explained the
correspondences.

    CALIBRATED (2)            essential matrix -- real baseline, triangulates
    UNCALIBRATED (3)          fundamental matrix -- real baseline
    PLANAR (4)                homography on a plane
    PANORAMIC (5)             PURE ROTATION -- no baseline at all
    PLANAR_OR_PANORAMIC (6)   ambiguous between the two
    DEGENERATE (1) / WATERMARK (7) / MULTIPLE (8)

Far-field matches on a distant skyline are exactly what lands in PANORAMIC: they
constrain orientation and give nothing to triangulate, so they cannot join two
reconstructions no matter how many inliers they carry. This is the question the
SIFT probe could only guess at.
"""
import sqlite3, sys
from collections import Counter, defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else 'colmap/merged/database.db'
A = {'0050', '0051', '0052', '0053', '0055'}          # hilltop
B = {'0057', '0058', '0059', '0060'}                  # cemetery
MAX_ID = 2147483647
CFG = {0: 'UNDEFINED', 1: 'DEGENERATE', 2: 'CALIBRATED', 3: 'UNCALIBRATED',
       4: 'PLANAR', 5: 'PANORAMIC', 6: 'PLANAR_OR_PANORAMIC', 7: 'WATERMARK',
       8: 'MULTIPLE'}
USABLE = {2, 3}                                       # has a real baseline

c = sqlite3.connect(DB)
name = {i: n for i, n in c.execute('select image_id, name from images')}

cfgs, inl, clip_pairs, per_img = Counter(), [], Counter(), defaultdict(int)
tot = 0
for pid, rows, cfg in c.execute('select pair_id, rows, config from two_view_geometries'):
    i2 = pid % MAX_ID
    i1 = (pid - i2) // MAX_ID
    n1, n2 = name.get(i1), name.get(i2)
    if not n1 or not n2:
        continue
    c1, c2 = n1[:4], n2[:4]
    if not ((c1 in A and c2 in B) or (c1 in B and c2 in A)):
        continue                                       # same-session pair
    tot += 1
    cfgs[cfg] += 1
    if rows >= 15:
        inl.append((rows, cfg, n1, n2))
    if cfg in USABLE and rows >= 30:
        clip_pairs[tuple(sorted((c1, c2)))] += 1
        per_img[n1] += 1; per_img[n2] += 1

print(f'cross-session pairs with a verified two-view geometry: {tot:,}\n')
print(f'{"config":24} {"count":>8}  {"share":>7}')
for k, v in cfgs.most_common():
    print(f'{CFG.get(k,k):24} {v:8,}  {100*v/max(tot,1):6.1f}%')

good = [x for x in inl if x[1] in USABLE and x[0] >= 30]
print(f'\nUSABLE links (essential/fundamental, >=30 inliers): {len(good):,}')
if good:
    good.sort(reverse=True)
    r = [g[0] for g in good]
    print(f'  inliers: max {r[0]}  median {r[len(r)//2]}  p90 {r[int(len(r)*0.1)]}')
    print(f'  distinct hilltop frames linked : {len([k for k in per_img if k[:4] in A])}')
    print(f'  distinct cemetery frames linked: {len([k for k in per_img if k[:4] in B])}')
    print('\n  clip-pair spread:')
    for (x, y), n in clip_pairs.most_common():
        print(f'    {x} <-> {y}: {n}')
    print('\n  strongest:')
    for rr, cf, n1, n2 in good[:6]:
        print(f'    {rr:4d} inliers  {CFG[cf]:12} {n1} <-> {n2}')
else:
    print('  none -- the sessions cannot be joined by these pairs')
