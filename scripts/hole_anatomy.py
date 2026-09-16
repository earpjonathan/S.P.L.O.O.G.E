#!/usr/bin/env python
"""Are the dark spots semi-transparent ground, or ground that isn't there?

If the near ground were merely too faint, quadrupling optical depth would push
accumulated alpha towards 1.  It moved 0.634 -> 0.721.  That is the signature
of rays that cross NO splats at all: an empty ray stays empty however opaque
you make the splats it never hits.  This separates the two.
"""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

SPLAT, CAMS = sys.argv[1], sys.argv[2]
W, H = 960, 720
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])
grid, lo, hi = terrain_grid(xyz)
agl = C[:,1]-height_at(grid,lo,hi,C[:,[0,2]])
good = (np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low = np.where(good & (agl<np.percentile(agl[good],20)))[0]
picks = low[np.linspace(0,len(low)-1,4).astype(int)]

allacc=[]; allcnt=[]
for k in picks:
    _, acc, cnt = render(xyz,S3,rgba,cams[k],W,H)
    allacc.append(acc[2*H//3:].ravel()); allcnt.append(cnt[2*H//3:].ravel())
acc=np.concatenate(allacc); cnt=np.concatenate(allcnt)

print(f"bottom third of {len(picks)} low-flight frames, {len(acc):,} pixels\n")
print(f"{'accumulated alpha':>20} {'share':>8}   {'median splats on ray':>21}")
bins=[(0,.05),(.05,.2),(.2,.5),(.5,.8),(.8,.95),(.95,1.01)]
for a,b in bins:
    m=(acc>=a)&(acc<b)
    if m.sum()==0: continue
    print(f"{a:>9.2f}-{b:<9.2f} {100*m.mean():>7.1f}%   {np.median(cnt[m]):>21.0f}")
print(f"\npixels crossing ZERO splats : {100*np.mean(cnt==0):.1f}%")
print(f"pixels crossing 1-2 splats  : {100*np.mean((cnt>0)&(cnt<3)):.1f}%")
print(f"median splats per pixel     : {np.median(cnt):.0f}")
op = rgba[:,3]
print(f"\nif every splat were fully opaque, those rays would still be empty:")
print(f"  share of bottom third with <3 splats on the ray: "
      f"{100*np.mean(cnt<3):.1f}%   (measured holes acc<0.5: {100*np.mean(acc<0.5):.1f}%)")
