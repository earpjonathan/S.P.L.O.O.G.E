#!/usr/bin/env python
"""SIFT + ratio test + RANSAC fundamental matrix between frames of two clips.
Verified inlier count is the practical test for 'can COLMAP register these
together' -- if cross-session frames do not match, extra flights cannot help.
"""
import sys, glob, itertools, numpy as np, cv2

sift = cv2.SIFT_create(nfeatures=8000)
cache = {}
def feats(p):
    if p not in cache:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        cache[p] = sift.detectAndCompute(img, None)
    return cache[p]

def pair_inliers(p1, p2):
    k1, d1 = feats(p1); k2, d2 = feats(p2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8: return 0
    bf = cv2.BFMatcher()
    mm = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in mm if m.distance < 0.8 * n.distance]
    if len(good) < 8: return 0
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.99)
    return 0 if mask is None else int(mask.sum())

clips = sys.argv[1:]
frames = {c: sorted(glob.glob(f'work/xmatch/{c}_*.jpg')) for c in clips}
ref = clips[0]
print(f'reference clip {ref} ({len(frames[ref])} frames)\n')
print(f'{"pair":<14}{"best":>7}{"median":>8}{"n>=30":>7}   verdict')
for c in clips[1:]:
    vals = [pair_inliers(f1, f2) for f1 in frames[ref] for f2 in frames[c]]
    vals = np.array(vals)
    strong = int((vals >= 30).sum())
    verdict = ('registers easily' if strong >= len(vals)*0.15 else
               'marginal' if strong > 0 else 'CANNOT register')
    print(f'{ref}-{c:<9}{vals.max():>7}{int(np.median(vals)):>8}{strong:>7}   {verdict}')
