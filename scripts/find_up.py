#!/usr/bin/env python
"""Robust scene-up from a RANSAC ground-plane fit to the COLMAP points.

Brush's `comment Vertical axis:` is an estimate and it is NOT always right:
on the 0026-only scene it was usable, on the 0026+0027 scene it came out
tilted far enough that neither +Y nor -Y was up after levelling. Deriving up
from the ground plane is checkable -- the cameras must end up ABOVE it.
"""
import numpy as np

def ground_normal(P, C, iters=2000, seed=0):
    """P: COLMAP points (N,3). C: camera centres (M,3). Returns unit up vector."""
    rng = np.random.default_rng(seed)
    # work on the central bulk; distant background biases the plane.
    # Subsample first: RANSAC over a million points is needlessly expensive
    # (and OOMs when every trial scores the full cloud).
    if len(P) > 60000:
        P = P[rng.choice(len(P), 60000, replace=False)]
    r = np.linalg.norm(P - np.median(P, 0), axis=1)
    Q = P[r < np.percentile(r, 80)]
    best_n, best_in = None, -1
    scale = np.percentile(np.linalg.norm(Q - Q.mean(0), axis=1), 50)
    tol = 0.05 * scale
    for _ in range(iters):
        i = rng.choice(len(Q), 3, replace=False)
        a, b, c = Q[i]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12: continue
        nrm = nrm / ln
        d = np.abs((Q - a) @ nrm)
        ni = int((d < tol).sum())
        if ni > best_in: best_in, best_n, best_a = ni, nrm, a
    # refine on inliers with a least-squares plane
    d = np.abs((Q - best_a) @ best_n)
    inl = Q[d < tol]
    cen = inl.mean(0)
    u, s, vt = np.linalg.svd(inl - cen)
    n = vt[-1]
    # orient so the cameras are above the ground
    if np.median((C - cen) @ n) < 0: n = -n
    return n / np.linalg.norm(n), best_in / len(Q)
