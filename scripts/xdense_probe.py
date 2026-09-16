#!/usr/bin/env python
"""Dense cross-SESSION probe: is there ANY frame pair that matches, and what
does it look like? Reports the top pairs by name and writes side-by-sides, so
a negative result can be inspected rather than just believed."""
import glob, numpy as np, cv2, os, sys
sift = cv2.SIFT_create(nfeatures=6000)
FL = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
cache = {}
def feats(p):
    if p not in cache:
        cache[p] = sift.detectAndCompute(cv2.imread(p, cv2.IMREAD_GRAYSCALE), None)
    return cache[p]
def inl(p1, p2):
    k1,d1 = feats(p1); k2,d2 = feats(p2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8: return 0
    mm = FL.knnMatch(d1, d2, k=2)
    g = [m for m,n in (x for x in mm if len(x)==2) if m.distance < 0.8*n.distance]
    if len(g) < 8: return 0
    a = np.float32([k1[m.queryIdx].pt for m in g]); b = np.float32([k2[m.trainIdx].pt for m in g])
    _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.99)
    return 0 if mask is None else int(mask.sum())

D='work/xdense'
old = sorted(glob.glob(f'{D}/OLD_s*.jpg'))
# control: how well do the OLD probes match each other at this spacing
ctrl = [inl(old[i], old[i+1]) for i in range(len(old)-1)]
print(f'CONTROL old-vs-old consecutive ({len(old)} probes over both clips): '
      f'median {int(np.median(ctrl))}  p90 {int(np.percentile(ctrl,90))}  max {max(ctrl)}')
os.makedirs('reports/xdense', exist_ok=True)
for c in sys.argv[1:]:
    new = sorted(glob.glob(f'{D}/{c}_s*.jpg'))
    best = []
    for a in old:
        for b in new:
            best.append((inl(a,b), a, b))
    best.sort(reverse=True, key=lambda x: x[0])
    v = np.array([x[0] for x in best])
    print(f'\n{c}: {len(v)} pairs  max {v.max()}  p99 {int(np.percentile(v,99))} '
          f'>=40 {int((v>=40).sum())}  >=80 {int((v>=80).sum())}')
    for k,(n,a,b) in enumerate(best[:3]):
        print(f'   #{k+1} {n:4d} inliers   {os.path.basename(a)}  <->  {os.path.basename(b)}')
        ia, ib = cv2.imread(a), cv2.imread(b)
        h = 480
        ia = cv2.resize(ia, (int(ia.shape[1]*h/ia.shape[0]), h))
        ib = cv2.resize(ib, (int(ib.shape[1]*h/ib.shape[0]), h))
        cv2.imwrite(f'reports/xdense/{c}_top{k+1}_{n}.jpg', np.hstack([ia, ib]))
