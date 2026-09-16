#!/usr/bin/env python
"""How BROAD is the overlap between two sessions, not just how strong.

A single lucky frame pair cannot join two models; a contiguous stretch of the
flight that sees shared GROUND can. So report, per probe, its best cross-session
match and where those inliers sat, then summarise what fraction of each session
has a usable (ground, >=80) link. Saves the full matrix so follow-ups need no
re-matching.
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
    k1, d1 = feats(p1); k2, d2 = feats(p2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8: return 0, np.nan
    mm = FL.knnMatch(d1, d2, k=2)
    g = [m for m, n in (x for x in mm if len(x) == 2) if m.distance < 0.8 * n.distance]
    if len(g) < 8: return 0, np.nan
    a = np.float32([k1[m.queryIdx].pt for m in g]); b = np.float32([k2[m.trainIdx].pt for m in g])
    _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.99)
    if mask is None: return 0, np.nan
    m = mask.ravel().astype(bool)
    if m.sum() == 0: return 0, np.nan
    return int(m.sum()), float(a[m][:, 1].mean() / 1440.0)

A, B, NA, NB = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def sample(s, n):
    fs = sorted(glob.glob(f'frames/{s}/*.jpg'))
    return [fs[i] for i in np.linspace(0, len(fs) - 1, n).astype(int)]
sa, sb = sample(A, NA), sample(B, NB)
N = np.zeros((NA, NB), int); Y = np.full((NA, NB), np.nan)
for i, x in enumerate(sa):
    for j, y in enumerate(sb):
        N[i, j], Y[i, j] = match(x, y)
np.savez('work/xsite_matrix.npz', N=N, Y=Y,
         sa=[os.path.basename(p) for p in sa], sb=[os.path.basename(p) for p in sb])

GROUND, STRONG = 0.45, 80          # y>=0.45 is below the horizon band; >=80 inliers
good = (N >= STRONG) & (Y >= GROUND)
print(f'{A} x {B}   {NA}x{NB} probes\n')
print(f'pairs >=%d inliers            : %d' % (STRONG, (N >= STRONG).sum()))
print(f'  of those, on GROUND (y>=%.2f): %d' % (GROUND, good.sum()))
print(f'{A} probes with a ground link : {good.any(1).sum()}/{NA} '
      f'({100*good.any(1).mean():.0f}%)')
print(f'{B} probes with a ground link : {good.any(0).sum()}/{NB} '
      f'({100*good.any(0).mean():.0f}%)')
def runs(mask):
    r, c = [], 0
    for v in mask:
        c = c + 1 if v else 0
        r.append(c)
    return max(r) if r else 0
print(f'longest contiguous {A} stretch : {runs(good.any(1))} probes')
print(f'longest contiguous {B} stretch : {runs(good.any(0))} probes')
print('\nclips involved in ground links:')
from collections import Counter
ca = Counter(os.path.basename(sa[i])[:4] for i in np.where(good.any(1))[0])
cb = Counter(os.path.basename(sb[j])[:4] for j in np.where(good.any(0))[0])
print(f'  {A}: {dict(ca)}\n  {B}: {dict(cb)}')
