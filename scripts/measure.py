#!/usr/bin/env python
"""LOW / MID bottom-third opacity for one splat file, same cameras every time."""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at
SPLAT, CAMS, REF = sys.argv[1], sys.argv[2], sys.argv[3]
W,H = 960,720
xyz,sc,rgba,rot = load_splat(SPLAT)
S3 = sigma_world(sc,rot)
cams = json.load(open(CAMS))
C=np.array([c['position'] for c in cams]); R=np.array([c['rotation'] for c in cams])
rx = load_splat(REF)[0]                       # camera picks from the reference
grid,lo,hi = terrain_grid(rx)
agl=C[:,1]-height_at(grid,lo,hi,C[:,[0,2]])
good=(np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low=np.where(good&(agl<np.percentile(agl[good],20)))[0]
mid=np.where(good&(agl>np.percentile(agl[good],45))&(agl<np.percentile(agl[good],55)))[0]
o=[];hl=[];mo=[]
for k in low[np.linspace(0,len(low)-1,3).astype(int)]:
    _,a,_=render(xyz,S3,rgba,cams[k],W,H); b=a[2*H//3:]
    o.append(b.mean()); hl.append(np.mean(b<0.5))
for k in mid[np.linspace(0,len(mid)-1,2).astype(int)]:
    _,a,_=render(xyz,S3,rgba,cams[k],W,H); mo.append(a[2*H//3:].mean())
print(f"{SPLAT.split('/')[-1]:<22} LOW op {np.mean(o):.3f}  "
      f"holes {100*np.mean(hl):.1f}%   MID op {np.mean(mo):.3f}", flush=True)
