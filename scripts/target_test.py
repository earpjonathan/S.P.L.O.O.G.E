#!/usr/bin/env python
"""How much of the see-through is reachable by opacity, and which splats own it."""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from flatten_ground import visible_surface
from near_diag import terrain_grid, height_at

SPLAT, CAMS = sys.argv[1], sys.argv[2]
W, H = 960, 720
xyz0, sc0, rgba0, rot0 = load_splat(SPLAT)
cams = json.load(open(CAMS))
C=np.array([c['position'] for c in cams]); R=np.array([c['rotation'] for c in cams])
RES=320
lo=np.percentile(xyz0[:,[0,2]],1,axis=0); hi=np.percentile(xyz0[:,[0,2]],99,axis=0)
surf,_=visible_surface(xyz0, rgba0[:,3], sc0, RES, lo, hi)
ix=np.clip(((xyz0[:,[0,2]]-lo)/(hi-lo)*RES).astype(int),0,RES-1)
h=xyz0[:,1]-surf[ix[:,0],ix[:,1]]
grid,glo,ghi=terrain_grid(xyz0)
agl=C[:,1]-height_at(grid,glo,ghi,C[:,[0,2]])
good=(np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low=np.where(good&(agl<np.percentile(agl[good],20)))[0]
mid=np.where(good&(agl>np.percentile(agl[good],45))&(agl<np.percentile(agl[good],55)))[0]
plow=low[np.linspace(0,len(low)-1,3).astype(int)]
pmid=mid[np.linspace(0,len(mid)-1,2).astype(int)]

def run(tag, mask, K):
    rgba=rgba0.copy()
    if mask is not None: rgba[mask,3]=1-(1-rgba[mask,3])**K
    S3=sigma_world(sc0,rot0)
    o=[];hl=[];mo=[]
    for k in plow:
        _,a,_=render(xyz0,S3,rgba,cams[k],W,H); b=a[2*H//3:]
        o.append(b.mean()); hl.append(np.mean(b<0.5))
    for k in pmid:
        _,a,_=render(xyz0,S3,rgba,cams[k],W,H); mo.append(a[2*H//3:].mean())
    n = len(xyz0) if mask is None else mask.sum()
    print(f"{tag:<34} {100*n/len(xyz0):>5.1f}% {np.mean(o):>8.3f} "
          f"{100*np.mean(hl):>8.1f}% {np.mean(mo):>8.3f}", flush=True)

print(f"{'config':<34} {'splats':>6} {'LOW op':>8} {'LOW holes':>9} {'MID op':>8}",
      flush=True)
run("original", None, 1)
run("ALL splats k=3 (ceiling)", np.ones(len(xyz0),bool), 3)
run("near surface -0.6..+0.15  k=3", (h>-0.6)&(h<0.15), 3)
run("near surface -0.6..+0.15  k=5", (h>-0.6)&(h<0.15), 5)
run("below surface only h<0    k=3", h<0.0, 3)
