#!/usr/bin/env python
"""Collapse the volumetric ground slab into an opaque surface shell.

Diagnosis this addresses (see README): on both terrain scenes ~48% of the
ground's opacity mass sits BELOW the visible surface, the 10->90% opacity
transition spans ~1.4x flight altitude, and the thin axis of a ground splat
sits a median 47 deg off the surface normal.  3DGS has no surface prior: from
grazing FPV views a diffuse semi-transparent slab and a crisp opaque surface
have identical loss, so the optimiser never concentrates it.  You only notice
when you fly close enough to be inside the slab.

Fix, entirely post-hoc (no retraining):
  1. visible surface = per-cell height where downward-accumulated opacity
     crosses 50% (not a percentile of centres -- that is biased by the slab)
  2. classify ground: near the surface, smooth cell, clear column above
  3. re-orient  : thin axis -> local terrain normal, major axis kept in-plane
  4. flatten    : thickness -> `--thin` x its own smallest scale
  5. snap       : pull centres onto the surface by `--snap`
  6. densify    : raise opacity so the collapsed shell composites opaque
  7. prune      : drop leftovers well under the surface

Splats outside the ground class are passed through untouched, so trees,
headstones and structures keep whatever the optimiser gave them.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, '/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat

def quat_to_rows(rot):
    """main.js convention: the ROWS of this matrix are the principal axes."""
    n = np.linalg.norm(rot, axis=1, keepdims=True); n[n == 0] = 1
    r = rot/n; w, x, y, z = r[:,0], r[:,1], r[:,2], r[:,3]
    R = np.empty((len(r), 3, 3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y+w*z); R[:,0,2]=2*(x*z-w*y)
    R[:,1,0]=2*(x*y-w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z+w*x)
    R[:,2,0]=2*(x*z+w*y);   R[:,2,1]=2*(y*z-w*x); R[:,2,2]=1-2*(x*x+y*y)
    return R

def rows_to_quat(R):
    """Inverse of quat_to_rows (Shepperd's method on the transpose)."""
    M = np.transpose(R, (0,2,1))              # rows-as-axes -> standard
    m00,m01,m02 = M[:,0,0],M[:,0,1],M[:,0,2]
    m10,m11,m12 = M[:,1,0],M[:,1,1],M[:,1,2]
    m20,m21,m22 = M[:,2,0],M[:,2,1],M[:,2,2]
    t = m00+m11+m22
    q = np.empty((len(M),4))
    c0 = t > 0
    s = np.sqrt(np.maximum(t[c0]+1.0,1e-12))*2
    q[c0,0]=0.25*s; q[c0,1]=(m21[c0]-m12[c0])/s
    q[c0,2]=(m02[c0]-m20[c0])/s; q[c0,3]=(m10[c0]-m01[c0])/s
    c1 = ~c0 & (m00>=m11) & (m00>=m22)
    s = np.sqrt(np.maximum(1.0+m00[c1]-m11[c1]-m22[c1],1e-12))*2
    q[c1,0]=(m21[c1]-m12[c1])/s; q[c1,1]=0.25*s
    q[c1,2]=(m01[c1]+m10[c1])/s; q[c1,3]=(m02[c1]+m20[c1])/s
    c2 = ~c0 & ~c1 & (m11>=m22)
    s = np.sqrt(np.maximum(1.0+m11[c2]-m00[c2]-m22[c2],1e-12))*2
    q[c2,0]=(m02[c2]-m20[c2])/s; q[c2,1]=(m01[c2]+m10[c2])/s
    q[c2,2]=0.25*s; q[c2,3]=(m12[c2]+m21[c2])/s
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(1.0+m22[c3]-m00[c3]-m11[c3],1e-12))*2
    q[c3,0]=(m10[c3]-m01[c3])/s; q[c3,1]=(m02[c3]+m20[c3])/s
    q[c3,2]=(m12[c3]+m21[c3])/s; q[c3,3]=0.25*s
    return q/np.linalg.norm(q,axis=1,keepdims=True)

def visible_surface(xyz, op, sc, res, lo, hi, minw=6):
    """Height at which downward-accumulated coverage reaches 50% per cell."""
    ss = np.sort(sc, axis=1)
    wgt = op * ss[:,1] * ss[:,2]                 # opacity x projected area
    ix = np.clip(((xyz[:,[0,2]]-lo)/(hi-lo)*res).astype(int), 0, res-1)
    cell = ix[:,0]*res + ix[:,1]
    order = np.lexsort((-xyz[:,1], cell))        # cell asc, height desc
    c, h, w = cell[order], xyz[order,1], wgt[order]
    bnd = np.searchsorted(c, np.arange(res*res+1))
    cw = np.cumsum(w)
    tot = np.zeros(res*res); surf = np.full(res*res, np.nan)
    for k in range(res*res):
        s0, e0 = bnd[k], bnd[k+1]
        if e0-s0 < minw: continue
        base = cw[s0-1] if s0 > 0 else 0.0
        acc = cw[s0:e0]-base
        tot[k] = acc[-1]
        j = np.searchsorted(acc, 0.5*acc[-1])
        surf[k] = h[s0+min(j, e0-s0-1)]
    surf = surf.reshape(res,res)
    m = np.isfinite(surf)
    if not m.all():
        from scipy.spatial import cKDTree
        gy,gx = np.mgrid[0:res,0:res]
        t = cKDTree(np.c_[gy[m],gx[m]])
        _,i = t.query(np.c_[gy[~m],gx[~m]])
        surf[~m] = surf[m][i]
    # light smoothing so normals are stable
    from scipy.ndimage import uniform_filter
    return uniform_filter(surf, 3), m.reshape(res,res)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--res', type=int, default=320)
    ap.add_argument('--band-lo', type=float, default=-0.35)
    ap.add_argument('--band-hi', type=float, default=0.05)
    ap.add_argument('--snap', type=float, default=0.85)
    ap.add_argument('--thin', type=float, default=1.0,
                    help='scale on the normal-axis thickness. v1 used 0.5 and '
                         'REGRESSED: thinning shortens every path through the '
                         'ground. Leave at 1.0 -- alignment, not thinness, is '
                         'what makes a downward ray hit the flat face.')
    ap.add_argument('--grow', type=float, default=1.25,
                    help='enlarge the two in-plane axes so the shell tiles '
                         'without gaps once the splats are coplanar')
    ap.add_argument('--boost', type=float, default=4.0,
                    help='optical-depth multiplier k, applied as '
                         'a -> 1-(1-a)^k, which multiplies tau by exactly k '
                         'in every ray direction')
    ap.add_argument('--prune-below', type=float, default=-0.30)
    ap.add_argument('--clear-above', type=float, default=0.12)
    a = ap.parse_args()

    xyz, sc, rgba, rot = load_splat(a.inp)
    op = rgba[:,3].copy(); rgb = rgba[:,:3].copy()
    N = len(xyz)
    lo = np.percentile(xyz[:,[0,2]],1,axis=0); hi = np.percentile(xyz[:,[0,2]],99,axis=0)
    surf, filled = visible_surface(xyz, op, sc, a.res, lo, hi)
    ix = np.clip(((xyz[:,[0,2]]-lo)/(hi-lo)*a.res).astype(int),0,a.res-1)
    sh = surf[ix[:,0],ix[:,1]]
    h = xyz[:,1]-sh

    gy_,gx_ = np.gradient(surf)
    cell = (hi-lo)/a.res
    rough = np.hypot(gy_/cell[0], gx_/cell[1])
    smooth = rough[ix[:,0],ix[:,1]] < 2.5

    # column above must be clear -> this is a top surface, not under canopy
    ss = np.sort(sc,axis=1); wgt = op*ss[:,1]*ss[:,2]
    cid = ix[:,0]*a.res+ix[:,1]
    above = np.bincount(cid, weights=wgt*(h > a.clear_above), minlength=a.res**2)
    total = np.bincount(cid, weights=wgt, minlength=a.res**2)
    openc = (above/np.maximum(total,1e-9)) < 0.25
    is_open = openc[cid]

    ground = (h > a.band_lo) & (h < a.band_hi) & smooth & is_open
    prune  = (h <= a.prune_below) & smooth & is_open
    print(f"{N:,} splats -> ground {ground.sum():,} ({100*ground.mean():.1f}%)  "
          f"prune {prune.sum():,} ({100*prune.mean():.1f}%)")

    # --- re-orient + flatten the ground splats ---------------------------
    g = np.where(ground)[0]
    R = quat_to_rows(rot[g])
    nrm = np.stack([-gy_[ix[g,0],ix[g,1]]/cell[0],
                    np.ones(len(g)),
                    -gx_[ix[g,0],ix[g,1]]/cell[1]], 1)
    nrm /= np.linalg.norm(nrm,axis=1,keepdims=True)
    big = np.argmax(sc[g],axis=1)
    maj = R[np.arange(len(g)), big, :]
    e1 = maj - (maj*nrm).sum(1,keepdims=True)*nrm          # major, in-plane
    n1 = np.linalg.norm(e1,axis=1,keepdims=True)
    bad = (n1[:,0] < 1e-6)
    e1[bad] = np.array([1.0,0,0]) - nrm[bad]*nrm[bad,0:1]
    e1 /= np.linalg.norm(e1,axis=1,keepdims=True)
    e2 = np.cross(nrm, e1)
    ssg = np.sort(sc[g],axis=1)
    newR = np.stack([e1, e2, nrm], 1)                      # rows = axes
    rot[g] = rows_to_quat(newR)
    sc[g] = np.stack([ssg[:,2]*a.grow, ssg[:,1]*a.grow, ssg[:,0]*a.thin], 1)
    xyz[g,1] -= h[g]*a.snap
    op[g] = 1.0 - (1.0 - op[g])**a.boost

    keep = ~prune
    xyz,sc,rgb,op,rot = xyz[keep],sc[keep],rgb[keep],op[keep],rot[keep]
    m = len(xyz)
    ss = np.sort(sc,axis=1); signif = op*ss[:,1]*ss[:,2]
    o = np.argsort(-signif)
    xyz,sc,rgb,op,rot = xyz[o],sc[o],rgb[o],op[o],rot[o]

    buf = np.empty((m,32),np.uint8)
    buf[:,0:12]  = xyz.astype('<f4').view(np.uint8).reshape(m,12)
    buf[:,12:24] = sc.astype('<f4').view(np.uint8).reshape(m,12)
    buf[:,24:27] = np.clip(rgb*255,0,255).astype(np.uint8)
    buf[:,27]    = np.clip(op*255,0,255).astype(np.uint8)
    nn = np.linalg.norm(rot,axis=1,keepdims=True); nn[nn==0]=1
    buf[:,28:32] = np.clip(rot/nn*128+128,0,255).astype(np.uint8)
    buf.tofile(a.out)
    print(f"wrote {a.out}: {m:,} splats ({m*32/1e6:.0f} MB)")

if __name__ == '__main__':
    main()
