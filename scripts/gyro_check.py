#!/usr/bin/env python
"""Rigorous COLMAP-vs-gyro rotation check: tests both quaternion conventions,
magnitude-only vs full-rotation-after-alignment."""
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
def axang(R):
    c=np.clip((np.trace(R)-1)/2,-1,1); th=np.arccos(c)
    if th<1e-9: return np.array([1.,0,0]),0.0
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(th))
    return v/np.linalg.norm(v), np.degrees(th)
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1)))

im=read_images(f'{sparse}/images.bin')
rows=sorted(((int(v['name'].split('_')[1].split('.')[0]), qvec2R(v['q'])) for v in im.values()
             if int(v['name'].split('_')[1].split('.')[0]) in G), key=lambda r:r[0])
idx=[];dCs=[];dGs=[]
for i in range(1,len(rows)):
    i0,Rc0=rows[i-1]; i1,Rc1=rows[i]
    idx.append(i1); dCs.append(Rc1@Rc0.T); dGs.append(qR(G[i1])@qR(G[i0]).T)
idx=np.array(idx)
print(f'{len(idx)} consecutive registered pairs, clip {clip}\n')

for label,gs in [('gyro as-is',dGs),('gyro inverted',[g.T for g in dGs])]:
    tC=np.array([axang(d)[1] for d in dCs]); tG=np.array([axang(g)[1] for g in gs])
    dm=np.abs(tC-tG)
    print(f'--- {label} ---')
    print(f'  MAGNITUDE only : median {np.median(dm):.3f}  p90 {np.percentile(dm,90):.3f}  '
          f'RMS {np.sqrt((dm**2).mean()):.3f}  >2deg {100*(dm>2).mean():.2f}%')
    m=tG>3.0
    X=np.array([axang(g)[0] for g in gs])[m]*tG[m,None]
    Y=np.array([axang(d)[0] for d in dCs])[m]*tC[m,None]
    U,S,Vt=np.linalg.svd(Y.T@X); A=U@np.diag([1,1,np.sign(np.linalg.det(U@Vt))])@Vt
    r=np.array([ang(dCs[i], A@gs[i]@A.T) for i in range(len(idx))])
    print(f'  FULL (aligned) : median {np.median(r):.3f}  p90 {np.percentile(r,90):.3f}  '
          f'RMS {np.sqrt((r**2).mean()):.3f}  >2deg {100*(r>2).mean():.2f}%')
    print(f'  axis fit quality: singular values {S/S[0]}')
    if label=='gyro as-is': np.save('work/resid_asis.npy',np.stack([idx,r]))
    else: np.save('work/resid_inv.npy',np.stack([idx,r]))
