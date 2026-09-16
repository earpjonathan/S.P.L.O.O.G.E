#!/usr/bin/env python
"""Cross-clip matchability, properly sampled: 18 frames spread over each clip,
all-pairs SIFT + ratio test + RANSAC. Reports the tail, since only overlapping
viewpoints should match at all -- the median is the noise floor, not the signal.
Includes a within-clip control so the noise floor and ceiling are both visible.
"""
import sys, glob, numpy as np, cv2
sift = cv2.SIFT_create(nfeatures=4000)
FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
cache = {}
def feats(p):
    if p not in cache:
        cache[p] = sift.detectAndCompute(cv2.imread(p, cv2.IMREAD_GRAYSCALE), None)
    return cache[p]
def inliers(p1, p2):
    k1,d1 = feats(p1); k2,d2 = feats(p2)
    if d1 is None or d2 is None or len(k1)<8 or len(k2)<8: return 0
    mm = FLANN.knnMatch(d1, d2, k=2)
    good=[m for m,n in (x for x in mm if len(x)==2) if m.distance < 0.8*n.distance]
    if len(good) < 8: return 0
    a=np.float32([k1[m.queryIdx].pt for m in good]); b=np.float32([k2[m.trainIdx].pt for m in good])
    _,mask = cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,3.0,0.99)
    return 0 if mask is None else int(mask.sum())

clips=sys.argv[1:]; ref=clips[0]
F={c:sorted(glob.glob(f'work/xmatch/{c}_s*.jpg')) for c in clips}
# within-clip control: consecutive sampled frames of the reference
ctrl=[inliers(F[ref][i],F[ref][i+1]) for i in range(len(F[ref])-1)]
print(f'CONTROL {ref} vs itself (consecutive samples): median {int(np.median(ctrl))}  max {max(ctrl)}')
print(f'\n{"pair":<12}{"max":>6}{"p90":>6}{"p75":>6}{"med":>6}{">=40":>6}{">=80":>6}  verdict')
for c in clips[1:]:
    v=np.array([inliers(f1,f2) for f1 in F[ref] for f2 in F[c]])
    s40=int((v>=40).sum()); s80=int((v>=80).sum())
    verdict=('strong overlap' if s80>=4 else 'usable overlap' if s40>=4
             else 'weak' if v.max()>=40 else 'no usable overlap')
    print(f'{ref}-{c:<7}{v.max():>6}{int(np.percentile(v,90)):>6}{int(np.percentile(v,75)):>6}'
          f'{int(np.median(v)):>6}{s40:>6}{s80:>6}  {verdict}   (n={len(v)})')
