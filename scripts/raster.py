#!/usr/bin/env python
"""Faithful CPU reimplementation of viewer/main.js's splat rasteriser.

The Claude browser pane cannot render the WebGL viewer, so verification has to
happen offline. This reproduces the shader exactly:

  Vrk      = 4 * Sigma_world           (generateTexture packs 4*sigma)
  J        = [[fx/z,0,-fx*x/z^2],[0,-fy/z,fy*y/z^2],[0,0,0]]
  cov2d    = J Sigma_cam J^T
  sigma_i  = sqrt(lambda_i)/2          (quad is +-2, frag is exp(-|p|^2))
  reject   if centre outside 1.2*w
  vColor  *= clamp(ndc_z + 1, 0, 1)    <-- the near fade under investigation

Toggles let us A/B the fade and the cull margin without touching the viewer.
"""
import json, sys
import numpy as np

def load_splat(path):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 32)
    xyz  = raw[:, 0:12].copy().view('<f4').reshape(-1, 3).astype(np.float64)
    sc   = raw[:, 12:24].copy().view('<f4').reshape(-1, 3).astype(np.float64)
    rgba = raw[:, 24:28].astype(np.float64) / 255.0
    rot  = (raw[:, 28:32].astype(np.float64) - 128.0) / 128.0
    return xyz, sc, rgba, rot

def sigma_world(sc, rot):
    """3D covariance, matching generateTexture's M = S*R then sigma = M^T M."""
    n = np.linalg.norm(rot, axis=1, keepdims=True); n[n == 0] = 1
    r = rot / n
    w, x, y, z = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    R = np.empty((len(r), 3, 3))
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y+w*z); R[:,0,2]=2*(x*z-w*y)
    R[:,1,0]=2*(x*y-w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z+w*x)
    R[:,2,0]=2*(x*z+w*y);   R[:,2,1]=2*(y*z-w*x); R[:,2,2]=1-2*(x*x+y*y)
    M = R * sc[:, :, None]                      # rows scaled, as in main.js
    S = np.einsum('nki,nkj->nij', M, M)         # M^T M
    return (4.0 * S).astype(np.float16).astype(np.float64)   # half-float packing

def render(xyz, S3, rgba, cam, W, H, znear=0.2, zfar=200.0,
           fade=True, margin=1.2, bg=0.0, dilate=0.0):
    C  = np.asarray(cam['position'], float)
    Rc = np.asarray(cam['rotation'], float)          # camera -> world
    s  = H / cam['height']
    fx, fy = cam['fx'] * s, cam['fy'] * s

    Xc = (xyz - C) @ Rc                              # world -> camera
    z  = Xc[:, 2]
    px = (2 * fx / W) * Xc[:, 0]
    py = -(2 * fy / H) * Xc[:, 1]
    pz = z * zfar / (zfar - znear) - zfar * znear / (zfar - znear)
    clip = margin * z
    keep = (z > 0) & (pz >= -clip) & (np.abs(px) <= clip) & (np.abs(py) <= clip)

    i = np.where(keep)[0]
    if len(i) == 0:
        return np.zeros((H, W, 3)), np.zeros((H, W)), np.zeros((H, W), np.int32)
    Xci, zi = Xc[i], z[i]

    Sc = np.einsum('ki,nkl,lj->nij', Rc, S3[i], Rc)  # Rc^T S Rc  (world->cam)
    # image-space Jacobian: x right, y DOWN, matching numpy row order. The
    # shader works in GL NDC (y up) and flips again when mapping to the
    # framebuffer; doing it once here keeps centre and ellipse consistent.
    J = np.zeros((len(i), 3, 3))
    J[:,0,0] = fx / zi;  J[:,0,2] = -fx * Xci[:,0] / zi**2
    J[:,1,1] = fy / zi;  J[:,1,2] = -fy * Xci[:,1] / zi**2
    cov = np.einsum('nik,nkl,njl->nij', J, Sc, J)[:, :2, :2]
    # Reference 3DGS applies a screen-space low-pass so no splat shrinks below
    # ~one pixel (cov2d[0][0] += 0.3).  main.js omits it, so edge-on splats
    # collapse to a line and stop covering pixels.  cov here is 4x the true
    # covariance (generateTexture packs 4*sigma), hence the factor.
    if dilate:
        cov = cov.copy()
        cov[:, 0, 0] += 4.0 * dilate
        cov[:, 1, 1] += 4.0 * dilate

    mid = (cov[:,0,0] + cov[:,1,1]) / 2
    rad = np.sqrt(np.maximum(((cov[:,0,0]-cov[:,1,1])/2)**2 + cov[:,0,1]**2, 0))
    l1, l2 = mid + rad, mid - rad
    ok = l2 > 0
    i, zi, l1, l2, cov = i[ok], zi[ok], l1[ok], l2[ok], cov[ok]
    Xci = Xci[ok]

    # principal axis, exactly as the shader derives it
    dv = np.stack([cov[:,0,1], l1 - cov[:,0,0]], 1)
    dvn = np.linalg.norm(dv, axis=1, keepdims=True); dvn[dvn == 0] = 1
    dv = dv / dvn
    # |majorAxis| = min(sqrt(2*lambda),1024); a quad vertex sits at
    # position.x = +-2 and lands position.x*|majorAxis|/2 pixels out, so
    # exp(-|position|^2) is a Gaussian of pixel std sqrt(lambda)/2, NOT
    # sqrt(2*lambda)/2.  Getting this wrong renders every splat sqrt(2) too
    # wide and overstates how opaque the scene is.
    R2 = 2.0 * np.sqrt(2.0)
    s1 = np.minimum(np.sqrt(2*l1), 1024.0) / R2      # px std along major
    s2 = np.minimum(np.sqrt(2*l2), 1024.0) / R2

    cx = fx * Xci[:,0] / zi + W / 2.0
    cy = fy * Xci[:,1] / zi + H / 2.0

    col = rgba[i, :3].copy()
    alpha = rgba[i, 3].copy()
    if fade:
        ndc = pz[i] / z[i]
        f = np.clip(ndc + 1.0, 0.0, 1.0)
        col = col * f[:, None]
        alpha = alpha * f

    order = np.argsort(zi)                            # front to back
    img = np.full((H, W, 3), bg, float)
    T   = np.ones((H, W))
    cnt = np.zeros((H, W), np.int32)      # splats actually covering each pixel
    ext = 2.0 * np.maximum(s1, s2) * 1.05

    for k in order:
        if alpha[k] <= 0.002:
            continue
        e = ext[k]
        x0 = max(int(cx[k]-e), 0); x1 = min(int(cx[k]+e)+1, W)
        y0 = max(int(cy[k]-e), 0); y1 = min(int(cy[k]+e)+1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        Tw = T[y0:y1, x0:x1]
        if Tw.max() < 0.004:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx + 0.5 - cx[k]; dy = yy + 0.5 - cy[k]
        u =  dx*dv[k,0] + dy*dv[k,1]                  # along major
        v = -dx*dv[k,1] + dy*dv[k,0]                  # along minor
        A = -(u*u/(2*s1[k]**2) + v*v/(2*s2[k]**2))
        m = A > -4.0
        if not m.any():
            continue
        a = np.zeros_like(A)
        a[m] = np.exp(A[m]) * alpha[k]
        contrib = Tw * a
        img[y0:y1, x0:x1] += contrib[:, :, None] * col[k]
        T[y0:y1, x0:x1] = Tw * (1 - a)
        cnt[y0:y1, x0:x1] += m
    return np.clip(img, 0, 1), 1.0 - T, cnt
