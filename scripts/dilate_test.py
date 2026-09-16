#!/usr/bin/env python
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from near_diag import terrain_grid, height_at

SPLAT, CAMS = sys.argv[1], sys.argv[2]
W, H = 480, 360
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams]); R = np.array([c['rotation'] for c in cams])
grid, lo, hi = terrain_grid(xyz)
agl = C[:,1] - height_at(grid, lo, hi, C[:,[0,2]])
good = (np.abs(R[:,1,0])<0.12)&(R[:,1,2]>-0.35)&(R[:,1,2]<0.05)
low  = np.where(good & (agl < np.percentile(agl[good],20)))[0]
mid  = np.where(good & (agl > np.percentile(agl[good],45))
                     & (agl < np.percentile(agl[good],55)))[0]
picks_low = low[np.linspace(0,len(low)-1,4).astype(int)]
picks_mid = mid[np.linspace(0,len(mid)-1,3).astype(int)]
print(f"{'dilate':>7} {'LOW bottom-3rd op':>19} {'holes':>7} | {'MID bottom-3rd op':>19} {'holes':>7}")
for d in (0.0, 0.1, 0.3, 0.6, 1.0):
    lo_o=[];lo_h=[];mi_o=[];mi_h=[]
    for k in picks_low:
        _,acc = render(xyz,S3,rgba,cams[k],W,H,dilate=d)
        b=acc[2*H//3:]; lo_o.append(b.mean()); lo_h.append(np.mean(b<0.5))
    for k in picks_mid:
        _,acc = render(xyz,S3,rgba,cams[k],W,H,dilate=d)
        b=acc[2*H//3:]; mi_o.append(b.mean()); mi_h.append(np.mean(b<0.5))
    print(f"{d:>7.2f} {np.mean(lo_o):>19.3f} {100*np.mean(lo_h):>6.1f}% | "
          f"{np.mean(mi_o):>19.3f} {100*np.mean(mi_h):>6.1f}%")
