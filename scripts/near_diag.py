#!/usr/bin/env python
"""Why is the ground see-through when flying low?

Reproduces the viewer's exact camera math (main.js getProjectionMatrix +
vertex shader) offline, and measures where the terrain actually falls
relative to the two hardcoded near-field thresholds:

  znear = 0.2   -> fade band: vColor is scaled by clamp(ndc_z+1,0,1),
                   which reaches 0 at z=0.100 and 1.0 at z=0.200.
  clip  = 1.2w  -> centre-only frustum reject, 20% margin beyond the edge.

Viewer frame: up is +Y (ply2splat maps gravity to (0,-1,0)).
"""
import json, sys
import numpy as np

ZNEAR, ZFAR = 0.2, 200.0

def load_splat(path, stride=1):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 32)[::stride]
    xyz = raw[:, 0:12].copy().view('<f4').reshape(-1, 3)
    sc  = raw[:, 12:24].copy().view('<f4').reshape(-1, 3)
    rgba = raw[:, 24:28]
    return xyz.astype(np.float64), sc.astype(np.float64), rgba

def terrain_grid(xyz, res=160, pct=8):
    """Heightfield: low percentile of splat Y per horizontal cell."""
    P = xyz[:, [0, 2]]; H = xyz[:, 1]
    lo = np.percentile(P, 1, axis=0); hi = np.percentile(P, 99, axis=0)
    ix = np.clip(((P - lo) / (hi - lo) * res).astype(int), 0, res - 1)
    flat = ix[:, 0] * res + ix[:, 1]
    order = np.argsort(flat, kind='stable')
    fs, hs = flat[order], H[order]
    b = np.searchsorted(fs, np.arange(res * res + 1))
    grid = np.full(res * res, np.nan)
    for c in range(res * res):
        s, e = b[c], b[c + 1]
        if e - s >= 8:
            grid[c] = np.percentile(hs[s:e], pct)
    grid = grid.reshape(res, res)
    m = np.isfinite(grid)
    if not m.all():                      # fill holes from nearest filled cell
        from scipy.spatial import cKDTree
        gy, gx = np.mgrid[0:res, 0:res]
        t = cKDTree(np.c_[gy[m], gx[m]])
        _, i = t.query(np.c_[gy[~m], gx[~m]])
        grid[~m] = grid[m][i]
    return grid, lo, hi

def height_at(grid, lo, hi, P):
    res = grid.shape[0]
    ix = np.clip(((P - lo) / (hi - lo) * res).astype(int), 0, res - 1)
    return grid[ix[:, 0], ix[:, 1]]

def main(splat, cams_json, label):
    xyz, sc, rgba = load_splat(splat, stride=1)
    cams = json.load(open(cams_json))
    print(f"\n{'='*66}\n{label}: {len(xyz):,} splats, {len(cams):,} cameras\n{'='*66}")

    grid, lo, hi = terrain_grid(xyz)
    C = np.array([c['position'] for c in cams])
    R = np.array([c['rotation'] for c in cams])
    fx = np.array([c['fx'] for c in cams]); fy = np.array([c['fy'] for c in cams])
    W = cams[0]['width']; H = cams[0]['height']

    gh = height_at(grid, lo, hi, C[:, [0, 2]])
    agl = C[:, 1] - gh
    print(f"scene extent XZ  : {np.round(hi-lo,2)}   Y span "
          f"{xyz[:,1].min():.2f} .. {xyz[:,1].max():.2f}")
    print(f"camera AGL       : median {np.median(agl):.3f}  p10 {np.percentile(agl,10):.3f} "
          f" p90 {np.percentile(agl,90):.3f}  min {agl.min():.3f}")
    print(f"znear 0.2 as a fraction of median AGL: {0.2/np.median(agl)*100:.0f}%")

    # --- ray-cast the frame onto the terrain, per camera ---
    # pixel grid: rows across the full frame, sample columns
    nrow, ncol = 24, 16
    py = (np.arange(nrow) + 0.5) / nrow          # 0 top .. 1 bottom
    px = (np.arange(ncol) + 0.5) / ncol
    gxp, gyp = np.meshgrid(px, py)
    gxp = gxp.ravel(); gyp = gyp.ravel()

    lowmask = agl < np.percentile(agl, 25)       # the "flying low" cameras
    idx = np.where(lowmask)[0]
    if len(idx) > 400:
        idx = idx[np.linspace(0, len(idx)-1, 400).astype(int)]
    print(f"\nanalysing {len(idx)} low-flight cameras (AGL < p25 = "
          f"{np.percentile(agl,25):.3f})")

    band_rows = {'top third': (0.0, 1/3), 'middle third': (1/3, 2/3),
                 'bottom third': (2/3, 1.0)}
    stats = {k: [] for k in band_rows}

    for i in idx:
        Rc = R[i]; Ci = C[i]
        # ray dirs in camera space, +z forward (viewer convention)
        dx = (gxp * W - W/2) / fx[i]
        dy = (gyp * H - H/2) / fy[i]
        dcam = np.stack([dx, dy, np.ones_like(dx)], 1)
        dworld = dcam @ Rc.T                       # R is cam->world
        dworld /= np.linalg.norm(dworld, axis=1, keepdims=True)
        # march along the ray to find the terrain crossing
        ts = np.linspace(0.02, 12.0, 500)
        hit = np.full(len(dworld), np.nan)
        for j in range(len(dworld)):
            p = Ci + np.outer(ts, dworld[j])
            th = height_at(grid, lo, hi, p[:, [0, 2]])
            below = p[:, 1] < th
            k = np.argmax(below) if below.any() else -1
            if k > 0:
                hit[j] = ts[k]
        # view-space depth of the hit  (z = t * dcam_z_component_normalised)
        zc = hit * (dcam[:, 2] / np.linalg.norm(dcam, axis=1))
        for name, (a, b) in band_rows.items():
            m = (gyp >= a) & (gyp < b) & np.isfinite(zc)
            if m.any():
                stats[name].append(zc[m])

    print(f"\n{'screen band':<14} {'ground hits':>11} {'median z':>9} {'p10 z':>8} "
          f"{'z<0.2':>7} {'z<0.1':>7}")
    print('-'*66)
    for name in band_rows:
        if not stats[name]:
            continue
        z = np.concatenate(stats[name])
        print(f"{name:<14} {len(z):>11,} {np.median(z):>9.3f} "
              f"{np.percentile(z,10):>8.3f} "
              f"{100*np.mean(z<ZNEAR):>6.1f}% {100*np.mean(z<0.1):>6.1f}%")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
