#!/usr/bin/env python
"""Is the ground a surface or a slab?

For open (non-tree) ground columns, build the vertical opacity profile from
the splats and measure how far it takes to go from transparent to opaque.
A true surface transitions over ~one splat thickness; a volumetric slab
ramps over a large fraction of flight altitude, and you can see into it.

Also characterises ground splat SHAPE: flatness (s_min/s_max) and whether
the thin axis actually points along the terrain normal.
"""
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat
from near_diag import terrain_grid, height_at

SPLAT, CAMS, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
xyz, sc, rgba, rot = load_splat(SPLAT)
op = rgba[:, 3]
cams = json.load(open(CAMS))
C = np.array([c['position'] for c in cams])

grid, lo, hi = terrain_grid(xyz)
th = height_at(grid, lo, hi, xyz[:, [0, 2]])
agl_cam = C[:,1] - height_at(grid, lo, hi, C[:,[0,2]])
unit = np.median(agl_cam)
print(f"\n{'='*70}\n{TAG}  ({len(xyz):,} splats)")
print(f"scene unit of interest: median camera AGL = {unit:.3f}\n{'='*70}")

# ---- 1. shape of the splats that sit at the ground surface --------------
h = xyz[:,1] - th                      # height above local terrain
s = np.sort(sc, axis=1)                # s[:,0] smallest .. s[:,2] largest
near = np.abs(h) < 0.05
# flat, open ground only: cells whose local height spread is small
res = grid.shape[0]
gh = np.gradient(grid)
rough = np.hypot(gh[0], gh[1])
ix = np.clip(((xyz[:,[0,2]] - lo)/(hi-lo)*res).astype(int), 0, res-1)
smooth = rough[ix[:,0], ix[:,1]] < np.percentile(rough, 40)
g = near & smooth
print(f"ground-surface splats (|h|<0.05, smooth terrain): {g.sum():,}")
flat = s[g,0]/np.maximum(s[g,2],1e-9)
print(f"  flatness s_min/s_max : median {np.median(flat):.3f}  "
      f"p10 {np.percentile(flat,10):.3f}  p90 {np.percentile(flat,90):.3f}")
print(f"  s_min (thickness)    : median {np.median(s[g,0]):.4f}  "
      f"= {np.median(s[g,0])/unit*100:.1f}% of flight altitude")
print(f"  s_max (extent)       : median {np.median(s[g,2]):.4f}")
print(f"  opacity              : median {np.median(op[g]):.3f}  "
      f"frac<0.5 {100*np.mean(op[g]<0.5):.1f}%")

# thin-axis orientation vs vertical
n = np.linalg.norm(rot, axis=1, keepdims=True); n[n==0]=1
r = rot/n; w,x,y,z = r[:,0],r[:,1],r[:,2],r[:,3]
R = np.empty((len(r),3,3))
R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y+w*z); R[:,0,2]=2*(x*z-w*y)
R[:,1,0]=2*(x*y-w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z+w*x)
R[:,2,0]=2*(x*z+w*y);   R[:,2,1]=2*(y*z-w*x); R[:,2,2]=1-2*(x*x+y*y)
axis = np.argmin(sc, axis=1)
thin = R[np.arange(len(R)), axis, :]        # rows of M are the scaled axes
ang = np.degrees(np.arccos(np.clip(np.abs(thin[g,1]), 0, 1)))
print(f"  thin axis vs vertical: median {np.median(ang):.1f} deg  "
      f"(0 = correctly aligned with the surface normal)")
print(f"    within 20 deg: {100*np.mean(ang<20):.1f}%   "
      f"beyond 60 deg: {100*np.mean(ang>60):.1f}%")

# ---- 2. vertical opacity profile through open ground --------------------
rng = np.random.default_rng(0)
cells = np.where(rough < np.percentile(rough,30))
pick = rng.choice(len(cells[0]), size=min(300,len(cells[0])), replace=False)
gy = lo[0] + (cells[0][pick]+0.5)/res*(hi[0]-lo[0])
gx = lo[1] + (cells[1][pick]+0.5)/res*(hi[1]-lo[1])

zs = np.linspace(-0.6, 0.3, 181)        # height relative to terrain
prof = np.zeros(len(zs)); ncol = 0
sig_h = np.maximum(np.mean(sc[:,:], axis=1), 1e-6)
for cx0, cz0 in zip(gy, gx):
    d2 = (xyz[:,0]-cx0)**2 + (xyz[:,2]-cz0)**2
    m = d2 < 0.09
    if m.sum() < 30: continue
    wgt = op[m]*np.exp(-0.5*d2[m]/np.maximum(sig_h[m],1e-4)**2)
    hh = h[m]
    prof += np.histogram(hh, bins=np.r_[zs, zs[-1]+ (zs[1]-zs[0])],
                         weights=wgt)[0]
    ncol += 1
prof /= max(ncol,1)
cum = np.cumsum(prof[::-1])[::-1]        # accumulate downward from above
cum = cum/cum.max() if cum.max()>0 else cum
def cross(f):
    i = np.argmax(cum >= f)
    return zs[::-1][len(zs)-1-i] if False else zs[i] if cum[i]>=f else np.nan
# cum is descending in index order (from top down); find where it crosses
order = np.argsort(-zs)                  # top to bottom
zt, ct = zs[order], cum[order]
def hit(f):
    k = np.argmax(ct >= f)
    return zt[k] if ct[k] >= f else np.nan
z10, z50, z90 = hit(0.10), hit(0.50), hit(0.90)
print(f"\nvertical opacity build-up over {ncol} open-ground columns "
      f"(height relative to terrain):")
print(f"  10% opaque at {z10:+.3f}   50% at {z50:+.3f}   90% at {z90:+.3f}")
print(f"  10->90 transition thickness: {abs(z90-z10):.3f} "
      f"= {abs(z90-z10)/unit*100:.0f}% of median flight altitude")
frac_below = prof[zs < z50].sum()/prof.sum() if prof.sum()>0 else 0
print(f"  opacity mass BELOW the 50% surface: {100*frac_below:.1f}%")
