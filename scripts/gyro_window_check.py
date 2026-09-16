#!/usr/bin/env python
"""Fit the camera<-imu alignment inside short windows. If poses are sound and
the gyro simply drifts, in-window residual is small and only the per-window
alignment rotates slowly over the clip."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_images, qvec2R
sparse=sys.argv[1]; clip=sys.argv[2]; W=int(sys.argv[3]) if len(sys.argv)>3 else 60
G={e['frame']: np.array(e['org_quat']) for e in json.load(open(f'gyro/cam_{clip}.json'))}
def qR(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
def ang(P,Q): return np.degrees(np.arccos(np.clip((np.trace(P.T@Q)-1)/2,-1,1)))
def proj(M):
    U,_,Vt=np.linalg.svd(M); return U@np.diag([1,1,np.sign(np.linalg.det(U@Vt))])@Vt
im=read_images(f'{sparse}/images.bin')
rows=sorted(((int(v['name'].split('_')[1].split('.')[0]), qvec2R(v['q'])) for v in im.values()
             if int(v['name'].split('_')[1].split('.')[0]) in G), key=lambda r:r[0])
idx=np.array([r[0] for r in rows]); Rc=np.array([r[1] for r in rows])
Rg=np.array([qR(G[i]) for i in idx])
allres=[]; wins=[]
for s in range(0,len(idx)-W,W):
    e=s+W; A=np.eye(3); B=np.eye(3)
    for _ in range(40):
        A=proj(sum(Rc[i]@(Rg[i]@B).T for i in range(s,e)))
        B=proj(sum((A@Rg[i]).T@Rc[i] for i in range(s,e)))
    r=np.array([ang(Rc[i],A@Rg[i]@B) for i in range(s,e)])
    allres.append(r); wins.append((idx[s],np.median(r),np.percentile(r,90)))
allres=np.concatenate(allres)
print(f'windowed (W={W} frames) alignment, {len(wins)} windows')
print(f'IN-WINDOW residual: median {np.median(allres):.3f}  p90 {np.percentile(allres,90):.3f}  '
      f'RMS {np.sqrt((allres**2).mean()):.3f} deg  >5deg {100*(allres>5).mean():.2f}%')
print('\nper-window median (frame -> deg):')
for f,m,p in wins: print(f'   {f:06d}  med {m:6.2f}  p90 {p:6.2f}')
