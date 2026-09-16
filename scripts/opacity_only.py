#!/usr/bin/env python
"""Ground opacity boost with NO geometry change, plus a visual A/B.

Every geometric intervention tried made this worse (thin shell 0.483, aligned
shell 0.421, vs 0.634 original) because the slab's THICKNESS is what is
currently providing surface coverage -- collapse it and the in-plane gaps show
through. Opacity is the only post-hoc lever that helps, and it plateaus.
"""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from flatten_ground import visible_surface
from near_diag import terrain_grid, height_at
from PIL import Image

K = 4.0
xyz, sc, rgba, rot = load_splat('viewer/aug2_web.splat')
cams = json.load(open('viewer/aug2_cameras.json'))
C=np.array([c['position'] for c in cams]); R=np.array([c['rotation'] for c in cams])
RES=320
lo=np.percentile(xyz[:,[0,2]],1,axis=0); hi=np.percentile(xyz[:,[0,2]],99,axis=0)
surf,_=visible_surface(xyz, rgba[:,3], sc, RES, lo, hi)
ix=np.clip(((xyz[:,[0,2]]-lo)/(hi-lo)*RES).astype(int),0,RES-1)
h=xyz[:,1]-surf[ix[:,0],ix[:,1]]
ground=(h>-0.6)&(h<0.15)
print(f"ground band splats: {ground.sum():,} ({100*ground.mean():.1f}%)", flush=True)

grid,glo,ghi=terrain_grid(xyz)
agl=C[:,1]-height_at(grid,glo,ghi,C[:,[0,2]])
good=(np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low=np.where(good&(agl<np.percentile(agl[good],20)))[0]
picks=low[np.linspace(0,len(low)-1,2).astype(int)]

W,Hh=640,480
S3=sigma_world(sc,rot)
rgba2=rgba.copy(); rgba2[ground,3]=1-(1-rgba2[ground,3])**K
rows=[]
for k in picks:
    tiles=[]
    for tag,rg in (('before',rgba),('after',rgba2)):
        img,acc,_=render(xyz,S3,rg,cams[k],W,Hh)
        b=acc[2*Hh//3:]
        print(f"  cam {k} {tag:<6} bottom-3rd {b.mean():.3f} "
              f"holes {100*np.mean(b<0.5):.1f}%", flush=True)
        tiles.append(np.clip(img+(1-acc)[:,:,None]*np.array([1,0.1,0.1]),0,1))
    rows.append(np.concatenate(tiles,1))
Image.fromarray((np.concatenate(rows,0)*255).astype(np.uint8)).save('work/opacity_ab.png')
print("left=before right=after -> work/opacity_ab.png", flush=True)
