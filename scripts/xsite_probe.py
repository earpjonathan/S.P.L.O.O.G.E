#!/usr/bin/env python
"""Can two SESSIONS be merged? Probe frame pairs across them for SIFT inliers.

Lessons from the May-vs-August attempt are baked in:
  * probe DENSELY -- 18 probes per clip once produced a false negative that 40
    per clip overturned;
  * always carry a WITHIN-session control, because the absolute inlier count
    means nothing without the noise floor and the ceiling for this scene;
  * report WHERE the inliers land. Far-field matches on a distant skyline
    constrain orientation and give no triangulation baseline, so they cannot
    join two models however many there are.
"""
import glob, os, sys
import numpy as np, cv2

sift = cv2.SIFT_create(nfeatures=6000)
FL = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
cache = {}

def feats(p):
    if p not in cache:
        cache[p] = sift.detectAndCompute(cv2.imread(p, cv2.IMREAD_GRAYSCALE), None)
    return cache[p]

def match(p1, p2):
    """-> (inlier count, mean y of inliers in image1 as a fraction of height)"""
    k1, d1 = feats(p1); k2, d2 = feats(p2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, None
    mm = FL.knnMatch(d1, d2, k=2)
    g = [m for m, n in (x for x in mm if len(x) == 2) if m.distance < 0.8 * n.distance]
    if len(g) < 8:
        return 0, None
    a = np.float32([k1[m.queryIdx].pt for m in g])
    b = np.float32([k2[m.trainIdx].pt for m in g])
    _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.99)
    if mask is None:
        return 0, None
    m = mask.ravel().astype(bool)
    if m.sum() == 0:
        return 0, None
    return int(m.sum()), float(a[m][:, 1].mean() / 1440.0)

def sample(setname, n):
    fs = sorted(glob.glob(f'frames/{setname}/*.jpg'))
    idx = np.linspace(0, len(fs) - 1, n).astype(int)
    return [fs[i] for i in idx]

A, B = sys.argv[1], sys.argv[2]
NA = int(sys.argv[3]) if len(sys.argv) > 3 else 55
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 55
sa, sb = sample(A, NA), sample(B, NB)
print(f'{A}: {len(sa)} probes over {len(glob.glob(f"frames/{A}/*.jpg"))} frames')
print(f'{B}: {len(sb)} probes over {len(glob.glob(f"frames/{B}/*.jpg"))} frames\n')

for nm, s in ((A, sa), (B, sb)):
    c = [match(s[i], s[i + 1])[0] for i in range(len(s) - 1)]
    print(f'CONTROL {nm} consecutive probes: median {int(np.median(c))} '
          f'p90 {int(np.percentile(c,90))} max {max(c)}')

res = []
for x in sa:
    for y in sb:
        n, ymean = match(x, y)
        res.append((n, ymean, x, y))
res.sort(reverse=True, key=lambda r: r[0])
v = np.array([r[0] for r in res])
print(f'\nCROSS {A} x {B}: {len(v)} pairs  max {v.max()}  p99 {int(np.percentile(v,99))}'
      f'  >=40 {int((v>=40).sum())}  >=80 {int((v>=80).sum())}')
for k, (n, ym, x, y) in enumerate(res[:5]):
    loc = 'n/a' if ym is None else (f'y={ym:.2f} ' +
          ('FAR-FIELD (horizon band)' if ym < 0.45 else 'lower frame (ground)'))
    print(f'   #{k+1} {n:4d} inliers  {os.path.basename(x)} <-> {os.path.basename(y)}  {loc}')
