#!/usr/bin/env python
"""Why the ground is see-through DOWNWARD but solid sideways, and what fixes it.

Accumulated alpha along a ray is 1 - prod(1-a_i).  The ground slab has a
vertical optical depth tau = sum -ln(1-a_i) of only ~1, so a ray looking
straight DOWN through it comes out ~60% opaque -- you see through.  A ray at a
grazing angle travels far enough inside the slab to cross many more splats, so
distant ground looks solid.  That is exactly the reported symptom: holes at the
bottom of the frame when low, solid everywhere else.

Thinning the slab (the obvious "make it flat" fix) makes this WORSE, because it
shortens every path.  The fix is to raise the optical depth instead:
    a -> 1 - (1-a)^k     multiplies tau by exactly k, for every ray direction.
"""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from flatten_ground import quat_to_rows, rows_to_quat, visible_surface
from near_diag import terrain_grid, height_at

SPLAT, CAMS = sys.argv[1], sys.argv[2]
W, H = 960, 720
xyz0, sc0, rgba0, rot0 = load_splat(SPLAT)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])

RES = 320
lo = np.percentile(xyz0[:,[0,2]],1,axis=0); hi = np.percentile(xyz0[:,[0,2]],99,axis=0)
surf,_ = visible_surface(xyz0, rgba0[:,3], sc0, RES, lo, hi)
ix = np.clip(((xyz0[:,[0,2]]-lo)/(hi-lo)*RES).astype(int),0,RES-1)
h = xyz0[:,1] - surf[ix[:,0],ix[:,1]]
gy_,gx_ = np.gradient(surf); cellsz=(hi-lo)/RES
rough = np.hypot(gy_/cellsz[0], gx_/cellsz[1])
smooth = rough[ix[:,0],ix[:,1]] < 2.5
ss = np.sort(sc0,axis=1); wgt = rgba0[:,3]*ss[:,1]*ss[:,2]
cid = ix[:,0]*RES+ix[:,1]
above = np.bincount(cid, weights=wgt*(h>0.12), minlength=RES**2)
total = np.bincount(cid, weights=wgt, minlength=RES**2)
is_open = ((above/np.maximum(total,1e-9))<0.25)[cid]
ground = (h>-0.35)&(h<0.05)&smooth&is_open
print(f"ground splats: {ground.sum():,} / {len(xyz0):,} ({100*ground.mean():.1f}%)",
      flush=True)

grid,glo,ghi = terrain_grid(xyz0)
agl = C[:,1]-height_at(grid,glo,ghi,C[:,[0,2]])
good = (np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low = np.where(good & (agl<np.percentile(agl[good],20)))[0]
mid = np.where(good & (agl>np.percentile(agl[good],45))
                    & (agl<np.percentile(agl[good],55)))[0]
plow = low[np.linspace(0,len(low)-1,3).astype(int)]
pmid = mid[np.linspace(0,len(mid)-1,2).astype(int)]

g = np.where(ground)[0]
Rr = quat_to_rows(rot0[g])
nrm = np.stack([-gy_[ix[g,0],ix[g,1]]/cellsz[0], np.ones(len(g)),
                -gx_[ix[g,0],ix[g,1]]/cellsz[1]],1)
nrm /= np.linalg.norm(nrm,axis=1,keepdims=True)
maj = Rr[np.arange(len(g)), np.argmax(sc0[g],axis=1), :]
e1 = maj-(maj*nrm).sum(1,keepdims=True)*nrm
n1 = np.linalg.norm(e1,axis=1,keepdims=True); bad=(n1[:,0]<1e-6)
e1[bad]=np.array([1.0,0,0])-nrm[bad]*nrm[bad,0:1]
e1/=np.linalg.norm(e1,axis=1,keepdims=True)
e2 = np.cross(nrm,e1); ssg=np.sort(sc0[g],axis=1)
qflat = rows_to_quat(np.stack([e1,e2,nrm],1))

print(f"\n{'config':<28} {'LOW op':>8} {'LOW holes':>10} {'MID op':>8} {'MID holes':>10}",
      flush=True)
def run(tag, xyz, sc, rgba, rot):
    S3 = sigma_world(sc, rot)
    o=[];hl=[];mo=[];mh=[]
    for k in plow:
        _,a = render(xyz,S3,rgba,cams[k],W,H); b=a[2*H//3:]
        o.append(b.mean()); hl.append(np.mean(b<0.5))
    for k in pmid:
        _,a = render(xyz,S3,rgba,cams[k],W,H); b=a[2*H//3:]
        mo.append(b.mean()); mh.append(np.mean(b<0.5))
    print(f"{tag:<28} {np.mean(o):>8.3f} {100*np.mean(hl):>9.1f}% "
          f"{np.mean(mo):>8.3f} {100*np.mean(mh):>9.1f}%", flush=True)

run("original", xyz0, sc0, rgba0, rot0)
for K in (2.0, 3.0, 4.0):
    xyz,sc,rgba,rot = xyz0.copy(), sc0.copy(), rgba0.copy(), rot0.copy()
    rgba[g,3] = 1-(1-rgba[g,3])**K
    run(f"opacity only  k={K:.0f}", xyz, sc, rgba, rot)
for K in (3.0,):
    xyz,sc,rgba,rot = xyz0.copy(), sc0.copy(), rgba0.copy(), rot0.copy()
    rgba[g,3] = 1-(1-rgba[g,3])**K
    rot[g] = qflat; sc[g] = np.stack([ssg[:,2],ssg[:,1],ssg[:,0]],1)
    xyz[g,1] -= h[g]*0.85
    run(f"snap+flatten+ k={K:.0f}", xyz, sc, rgba, rot)
