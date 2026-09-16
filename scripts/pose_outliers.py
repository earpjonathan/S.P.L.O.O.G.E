#!/usr/bin/env python
"""Per-frame COLMAP-vs-gyro rotation residual, after solving the fixed
camera<-imu alignment A (relative rotations are conjugated by A, so comparing
dC to dG directly is wrong -- it injects error of order the inter-frame angle)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_images, qvec2R

sparse = sys.argv[1] if len(sys.argv) > 1 else 'colmap/hedge26/sparse/0'
clip   = sys.argv[2] if len(sys.argv) > 2 else '0026'
G = {e['frame']: np.array(e['org_quat']) for e in json.load(open(f'gyro/cam_{clip}.json'))}

def qR(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
def axang(R):
    c=np.clip((np.trace(R)-1)/2,-1,1); th=np.arccos(c)
    if th<1e-8: return np.array([1.,0,0]),0.0
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(th))
    return v,np.degrees(th)
def ang(A,B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1)))

im=read_images(f'{sparse}/images.bin')
rows=sorted(((int(v['name'].split('_')[1].split('.')[0]), qvec2R(v['q'])) for v in im.values()
             if int(v['name'].split('_')[1].split('.')[0]) in G), key=lambda r:r[0])
print(f'matched {len(rows)} frames to gyro')

P=[]
for i in range(1,len(rows)):
    i0,Rc0=rows[i-1]; i1,Rc1=rows[i]
    dC=Rc1@Rc0.T; dG=qR(G[i1])@qR(G[i0]).T
    aC,tC=axang(dC); aG,tG=axang(dG)
    P.append((i1,dC,dG,aC,tC,aG,tG))

# magnitude-only check (alignment-free)
tC=np.array([p[4] for p in P]); tG=np.array([p[6] for p in P])
print(f'\n[magnitude only]  r={np.corrcoef(tC,tG)[0,1]:.4f}  '
      f'RMS diff={np.sqrt(((tC-tG)**2).mean()):.3f} deg')

# solve A by Kabsch on rotation axes (weight by angle; skip ill-conditioned small rotations)
m=tG>2.0
X=np.array([p[5] for p in P])[m]*tG[m,None]   # gyro axes, weighted
Y=np.array([p[3] for p in P])[m]*tC[m,None]   # colmap axes, weighted
U,S,Vt=np.linalg.svd(Y.T@X)
d=np.sign(np.linalg.det(U@Vt))
A=U@np.diag([1,1,d])@Vt
print(f'solved alignment A on {m.sum()} pairs')

res=np.array([ang(p[1], A@p[2]@A.T) for p in P])
print(f'\n[after alignment]  median {np.median(res):.3f}  p90 {np.percentile(res,90):.3f}  '
      f'RMS(all) {np.sqrt((res**2).mean()):.3f} deg')
bad=res>2.0
print(f'outliers >2deg: {bad.sum()} / {len(res)} ({100*bad.mean():.2f}%)')
ok=~bad
print(f'on the {100*ok.mean():.1f}% within 2deg: RMS {np.sqrt((res[ok]**2).mean()):.3f} deg')
idx=np.array([p[0] for p in P])
for j in np.argsort(-res)[:10]:
    print(f'   frame {idx[j]:06d}  {res[j]:8.2f} deg')
np.save('work/pose_resid.npy', np.stack([idx,res]))
