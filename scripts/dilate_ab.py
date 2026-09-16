#!/usr/bin/env python
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

SPLAT, CAMS = sys.argv[1], sys.argv[2]
W, H = int(sys.argv[3]), int(sys.argv[4])
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])
grid, lo, hi = terrain_grid(xyz)
agl = C[:,1] - height_at(grid, lo, hi, C[:,[0,2]])
good = (np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low = np.where(good & (agl < np.percentile(agl[good],20)))[0]
picks = low[np.linspace(0,len(low)-1,3).astype(int)]
print(f"{SPLAT.split('/')[-1]} at {W}x{H} (focal scale {H/1440:.2f} of training)",
      flush=True)
for d in (0.0, 0.3):
    o=[];hl=[]
    for k in picks:
        _,acc = render(xyz,S3,rgba,cams[k],W,H,dilate=d)
        b=acc[2*H//3:]; o.append(b.mean()); hl.append(np.mean(b<0.5))
    print(f"  dilate {d:.1f}: bottom-3rd opacity {np.mean(o):.3f}  "
          f"holes<0.5 {100*np.mean(hl):.1f}%", flush=True)
