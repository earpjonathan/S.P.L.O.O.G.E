#!/usr/bin/env python
"""All-pairs cross-clip matchability matrix, with a within-clip control per
clip so the noise floor is explicit. Answers 'which of these clips can COLMAP
actually join into one reconstruction'."""
import sys, glob, itertools, numpy as np, cv2
d=sys.argv[1]; clips=sys.argv[2:]
sift=cv2.SIFT_create(nfeatures=4000)
FL=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=32))
cache={}
def feats(p):
    if p not in cache: cache[p]=sift.detectAndCompute(cv2.imread(p,cv2.IMREAD_GRAYSCALE),None)
    return cache[p]
def inl(p1,p2):
    k1,d1=feats(p1); k2,d2=feats(p2)
    if d1 is None or d2 is None or len(k1)<8 or len(k2)<8: return 0
    mm=FL.knnMatch(d1,d2,k=2)
    g=[m for m,n in (x for x in mm if len(x)==2) if m.distance<0.8*n.distance]
    if len(g)<8: return 0
    a=np.float32([k1[m.queryIdx].pt for m in g]); b=np.float32([k2[m.trainIdx].pt for m in g])
    _,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,3.0,0.99)
    return 0 if mask is None else int(mask.sum())
F={c:sorted(glob.glob(f'{d}/{c}_s*.jpg')) for c in clips}
print('within-clip control (consecutive samples):')
for c in clips:
    v=[inl(F[c][i],F[c][i+1]) for i in range(len(F[c])-1)]
    print(f'   {c}: median {int(np.median(v)):>4}  max {max(v):>4}  '
          f'(low median = fast flight, little self-overlap)')
print(f'\n{"pair":<14}{"max":>6}{"p90":>6}{">=40":>6}{">=80":>6}  verdict')
res={}
for a,b in itertools.combinations(clips,2):
    v=np.array([inl(x,y) for x in F[a] for y in F[b]])
    s40=int((v>=40).sum()); s80=int((v>=80).sum()); res[(a,b)]=(v.max(),s40,s80)
    verd=('strong' if s80>=4 else 'usable' if s40>=4 else 'weak' if v.max()>=40 else 'none')
    print(f'{a}-{b:<9}{v.max():>6}{int(np.percentile(v,90)):>6}{s40:>6}{s80:>6}  {verd}')
