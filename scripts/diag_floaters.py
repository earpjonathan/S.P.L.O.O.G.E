#!/usr/bin/env python
"""Characterise the haze/'cloud' splats: height above terrain, size, opacity,
colour, and distance to the nearest COLMAP-triangulated point."""
import sys, os, numpy as np
sys.path.insert(0,'scripts')
from scipy.spatial import cKDTree
from colmap_io import read_points3D, read_images, qvec2R
from ply2splat import read_ply, rot_between

splatf = sys.argv[1] if len(sys.argv)>1 else 'viewer/site.splat'
sparse = sys.argv[2] if len(sys.argv)>2 else 'colmap/hedge26/sparse/0'
plyf   = sys.argv[3] if len(sys.argv)>3 else 'site_out/site_30000.ply'

d = np.fromfile(splatf, np.uint8).reshape(-1,32)
pos = d[:,0:12].copy().view('<f4').reshape(-1,3)
sc  = d[:,12:24].copy().view('<f4').reshape(-1,3)
rgb = d[:,24:27].astype(np.float32)/255.0
op  = d[:,27].astype(np.float32)/255.0
N=len(pos); print(f'{N:,} splats from {splatf}')

# rebuild the exact normalising transform ply2splat applied
_,_,vax = read_ply(plyf)
Rlev,_ = rot_between(np.asarray(vax,float), np.array([0.,-1.,0.]))
im = read_images(f'{sparse}/images.bin')
Cs = np.array([(-qvec2R(v['q']).T@v['t']) for v in im.values()]) @ Rlev.T
c  = Cs.mean(0); Cs = Cs - c
r90= np.percentile(np.linalg.norm(Cs,axis=1),90); sfac = 3.0/r90
Cs *= sfac
_,pxyz,_,perr,ptl = read_points3D(f'{sparse}/points3D.bin')
P = (pxyz @ Rlev.T - c) * sfac
print(f'{len(P):,} COLMAP points; transform sfac={sfac:.4f}')

up = -pos[:,1]                      # levelled so world-up maps to -Y
upC = -Cs[:,1]
print(f'\ncamera height:  p5 {np.percentile(upC,5):.2f}  median {np.median(upC):.2f}  p95 {np.percentile(upC,95):.2f}')
print(f'colmap pt height: p5 {np.percentile(-P[:,1],5):.2f}  median {np.median(-P[:,1]):.2f}  p95 {np.percentile(-P[:,1],95):.2f}')
print(f'splat height:   p5 {np.percentile(up,5):.2f}  median {np.median(up):.2f}  '
      f'p95 {np.percentile(up,95):.2f}  max {up.max():.2f}')

tree = cKDTree(P)
step = max(1, N//400_000)
idx = np.arange(0,N,step)
dist,_ = tree.query(pos[idx], k=1, workers=-1)
size = np.sort(sc[idx],axis=1)[:,2]          # largest axis
print(f'\ndistance to nearest COLMAP point (sample {len(idx):,}):')
for q in [50,75,90,95,99]:
    print(f'   p{q}: {np.percentile(dist,q):.3f}')

far = dist > np.percentile(dist,90)
near= dist <= np.percentile(dist,50)
def stats(mask,label):
    print(f'  {label:<22} n={mask.sum():>7,}  height {np.median(up[idx][mask]):6.2f}  '
          f'size {np.median(size[mask]):.4f}  opacity {np.median(op[idx][mask]):.3f}  '
          f'rgb {np.round(rgb[idx][mask].mean(0),2)}  blueness '
          f'{np.median(rgb[idx][mask][:,2]-rgb[idx][mask][:,:2].mean(1)):+.3f}')
print()
stats(near,'near geometry (p50)')
stats(far ,'far from geometry(p90)')
np.save('work/floater_diag.npy', np.stack([dist,up[idx],size,op[idx]]))
