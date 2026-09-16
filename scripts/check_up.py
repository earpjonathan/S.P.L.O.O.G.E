#!/usr/bin/env python
"""Score a candidate scene-up: a correct up puts every camera ABOVE the terrain
and makes splat AGL tightly concentrated at 0 (real surface) with a thin high
tail (sky). A wrong up smears AGL and sinks the cameras underground."""
import numpy as np, sys
sys.path.insert(0,'scripts')
from colmap_io import read_points3D, read_images, qvec2R
from ply2splat import read_ply, rot_between
from agl import terrain_field, agl
ply,sp = sys.argv[1], sys.argv[2]
cands = {'brush': None}
for extra in sys.argv[3:]:
    k,v = extra.split('=',1); cands[k]=np.array([float(t) for t in v.split(',')])
v_,n_,vax = read_ply(ply)
xyz=np.stack([v_['x'],v_['y'],v_['z']],1)
im=read_images(f'{sp}/images.bin')
C0=np.array([-qvec2R(x['q']).T@x['t'] for x in im.values()])
_,P0,_,_,_=read_points3D(f'{sp}/points3D.bin')
print(f'{"up":>8}{"camAGL p5":>11}{"camAGL med":>12}{"below%":>8}{"splat p95":>11}{"p99":>9}{"sep":>8}')
for name,u in cands.items():
    uu = np.asarray(vax if u is None else u,float)
    R,_=rot_between(uu, np.array([0.,-1.,0.]))
    X=xyz@R.T; C=C0@R.T; P=P0@R.T
    for sgn in (+1,-1):
        ch=C[:,[0,2]]; lo=ch.min(0); hi=ch.max(0); pad=0.6*(hi-lo); lo-=pad; hi+=pad
        g,tlo,thi=terrain_field(P[:,[0,2]], sgn*P[:,1], res=192, pct=60, extent=(lo,hi))
        Ac=agl(ch, sgn*C[:,1], g,tlo,thi)
        A=agl(X[:,[0,2]], sgn*X[:,1], g,tlo,thi)
        ins=((X[:,[0,2]]>=lo)&(X[:,[0,2]]<=hi)).all(1)
        p95=np.percentile(A[ins],95); p99=np.percentile(A[ins],99)
        print(f'{name+("+Y" if sgn>0 else "-Y"):>8}{np.percentile(Ac,5):>11.2f}'
              f'{np.median(Ac):>12.2f}{100*np.mean(Ac<0):>7.0f}%{p95:>11.2f}{p99:>9.2f}'
              f'{p99/max(p95,1e-6):>8.1f}')
