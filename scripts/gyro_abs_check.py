#!/usr/bin/env python
"""Absolute-orientation check: solve Rc_i ~= A Rg_i B (two-sided Procrustes,
alternating), then report per-frame angular residual. Well-conditioned for an
orbit, unlike aligning near-collinear inter-frame rotation axes."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_images, qvec2R

sparse=sys.argv[1]; clip=sys.argv[2]
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
print(f'{len(idx)} registered frames, clip {clip}')

# axis spread of ABSOLUTE gyro orientations -> conditioning of this fit
allax=np.concatenate([Rg[:,:,a] for a in range(3)],0)
print(f'absolute-orientation spread (SVD of stacked axes): '
      f'{np.linalg.svd(Rg[:,:,2]-Rg[:,:,2].mean(0))[1]/np.linalg.svd(Rg[:,:,2]-Rg[:,:,2].mean(0))[1][0]}')

A=np.eye(3); B=np.eye(3)
for it in range(60):
    A=proj(sum(Rc[i]@(Rg[i]@B).T for i in range(len(idx))))
    B=proj(sum((A@Rg[i]).T@Rc[i] for i in range(len(idx))))
res=np.array([ang(Rc[i], A@Rg[i]@B) for i in range(len(idx))])
print(f'\nABSOLUTE residual: median {np.median(res):.3f}  p90 {np.percentile(res,90):.3f}  '
      f'RMS {np.sqrt((res**2).mean()):.3f} deg   >5deg {100*(res>5).mean():.2f}%')

# is the residual drift (slow trend) or noise?
t=(idx-idx[0])/float(idx[-1]-idx[0])
sl,ic=np.polyfit(t,res,1)
det=res-(sl*t+ic)
print(f'linear trend over clip: {ic:.2f} -> {sl+ic:.2f} deg (slope {sl:.2f} deg/clip)  '
      f'= gyro integration drift')
print(f'detrended residual: median {np.median(np.abs(det)):.3f}  RMS {np.sqrt((det**2).mean()):.3f} deg')
worst=np.argsort(-np.abs(det))[:8]
print('worst frames after detrend:')
for j in worst: print(f'   {idx[j]:06d}  {res[j]:7.2f} deg (detrended {det[j]:+.2f})')
np.save('work/abs_resid.npy', np.stack([idx,res,det]))
