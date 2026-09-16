#!/usr/bin/env python
"""Terrain height field from COLMAP points + above-ground-level for splats.

Distance-to-nearest-COLMAP-point is a bad floater test on this scene:
textureless dirt yields few SIFT features, so *real* ground splats there are
far from any triangulated point and (being smooth) large. AGL separates them
properly - real surface splats sit at terrain height whatever the texture.
"""
import numpy as np

def terrain_field(P_up, P_h, res=96, pct=40, minpts=4, extent=None):
    """P_up: (N,2) horizontal coords of COLMAP points; P_h: their heights.

    `extent` ((lo2,hi2)) should be the FLIGHT area, not the full point cloud:
    the cloud includes distant city and hills, which stretch the grid until
    each cell spans a whole hillside and the terrain estimate is meaningless
    (cameras came out *below* ground). Points outside the extent are ignored.
    """
    if extent is not None:
        lo, hi = np.asarray(extent[0], float), np.asarray(extent[1], float)
        m = ((P_up >= lo) & (P_up <= hi)).all(1)
        P_up, P_h = P_up[m], P_h[m]
    else:
        lo = np.percentile(P_up, 1, axis=0); hi = np.percentile(P_up, 99, axis=0)
        pad = 0.05 * (hi - lo); lo -= pad; hi += pad
    ix = np.clip(((P_up - lo) / (hi - lo) * res).astype(int), 0, res - 1)
    flat = ix[:, 0] * res + ix[:, 1]
    order = np.argsort(flat, kind='stable')
    fs, hs = flat[order], P_h[order]
    bounds = np.searchsorted(fs, np.arange(res * res + 1))
    grid = np.full(res * res, np.nan)
    for cell in range(res * res):
        s, e = bounds[cell], bounds[cell + 1]
        if e - s >= minpts:
            grid[cell] = np.percentile(hs[s:e], pct)
    grid = grid.reshape(res, res)
    # fill empty cells from the nearest filled one
    valid = np.isfinite(grid)
    if not valid.all():
        from scipy.spatial import cKDTree
        gy, gx = np.mgrid[0:res, 0:res]
        vpts = np.stack([gy[valid], gx[valid]], 1)
        qpts = np.stack([gy[~valid], gx[~valid]], 1)
        _, j = cKDTree(vpts).query(qpts, k=1)
        grid[~valid] = grid[valid][j]
    return grid, lo, hi

def agl(xyz_up, xyz_h, grid, lo, hi):
    res = grid.shape[0]
    ix = np.clip(((xyz_up - lo) / (hi - lo) * res).astype(int), 0, res - 1)
    return xyz_h - grid[ix[:, 0], ix[:, 1]]
