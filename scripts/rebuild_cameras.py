#!/usr/bin/env python
"""Rebuild viewer/cameras.json for the May site scene after it was clobbered.

Nothing is guessed. ply2splat's viewer transform is fully determined by one
unknown, the levelling rotation Rlev (from --up); the centre and scale are then
derived from the COLMAP cameras themselves:

    Cn = (Rlev @ C - c) * sfac,   c = mean(Rlev @ C),   sfac = 3 / p90(|.|)
    Rn = Rlev @ R_c2w

site_traj.json holds Cn for resampled times, so Rlev is recoverable by fitting a
similarity transform between the COLMAP camera centres and those positions. The
resampled grid is t0 + k*dt with the source frames at idx/fps, so wherever those
coincide the stored sample IS the transformed COLMAP centre exactly (np.interp
returns the node value at a node), giving an exact correspondence rather than an
interpolated approximation.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from colmap_io import read_images, read_cameras, qvec2R

MODEL, TRAJ, OUT, STRIDE, FPS = ('colmap/both/sparse/0', 'viewer/site_traj.json',
                                 'viewer/cameras.json', 40, 50.0)

traj = json.load(open(TRAJ))
im = read_images(MODEL + '/images.bin')
cm = read_cameras(MODEL + '/cameras.bin')

# ---- exact correspondences ----
byclip = {c['name']: c for c in traj['clips']}
C_col, P_view = [], []
for k in im:
    nm = os.path.splitext(im[k]['name'])[0]
    clip, idx = nm.split('_')[0], int(nm.split('_')[1])
    c = byclip.get(clip)
    if c is None:
        continue
    kk = (idx / FPS - c['t0']) / c['dt']
    r = round(kk)
    if abs(kk - r) > 1e-6 or not (0 <= r < c['n']):
        continue
    C_col.append(-qvec2R(im[k]['q']).T @ im[k]['t'])
    P_view.append(c['pos'][3*r:3*r+3])   # pos is flat xyz
C_col, P_view = np.array(C_col), np.array(P_view)
print(f'{len(C_col)} exact frame/sample coincidences')

# ---- Umeyama similarity fit ----
mA, mB = C_col.mean(0), P_view.mean(0)
A, B = C_col - mA, P_view - mB
U, S, Vt = np.linalg.svd(B.T @ A / len(A))
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R = U @ D @ Vt
s = (S * np.diag(D)).sum() / (A ** 2).sum() * len(A)
b = mB - s * R @ mA
resid = np.abs((C_col @ R.T) * s + b - P_view).max()
print(f'fitted  scale {s:.6f}  b {np.round(b,4)}  max residual {resid:.2e}')
# The JSON stores positions rounded to 4 decimals, so ~5e-5 is the quantisation
# floor, not a transform error. Anything an order of magnitude above it is real.
if resid > 5e-4:
    raise SystemExit(f'residual {resid:.2e} too large -- transform not recovered')

# Second, independent estimate of Rlev from the stored camera ROTATIONS, which
# is the estimator recover_transform itself uses: Rlev = mean(Rn @ Rc2w.T),
# re-orthogonalised. Averaging 499 of them beats the 4-decimal position rounding.
from scipy.spatial.transform import Rotation
M = np.zeros((3, 3))
for k in im:
    nm = os.path.splitext(im[k]['name'])[0]
    clip, idx = nm.split('_')[0], int(nm.split('_')[1])
    cc = byclip.get(clip)
    if cc is None:
        continue
    kk = (idx / FPS - cc['t0']) / cc['dt']; r = round(kk)
    if abs(kk - r) > 1e-6 or not (0 <= r < cc['n']):
        continue
    Rn = Rotation.from_quat(cc['cam_quat'][4*r:4*r+4]).as_matrix()   # scipy: x,y,z,w
    M += Rn @ qvec2R(im[k]['q'])          # R_c2w.T == qvec2R(q)
U2, _, Vt2 = np.linalg.svd(M / len(C_col))
R2 = U2 @ Vt2
if np.linalg.det(R2) < 0:
    U2[:, -1] *= -1; R2 = U2 @ Vt2
ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ R2) - 1) / 2, -1, 1)))
print(f'position-fit vs rotation-fit Rlev disagree by {ang:.6f} deg')
R = R2

# ---- rebuild exactly as ply2splat would, deriving c/sfac from the cameras ----
ks = sorted(im, key=lambda k: im[k]['name'])
C = np.array([-qvec2R(im[k]['q']).T @ im[k]['t'] for k in ks])
Cl = C @ R.T
c = Cl.mean(0)
sfac = 3.0 / np.percentile(np.linalg.norm(Cl - c, axis=1), 90)
print(f'derived scale {sfac:.6f}  b {np.round(-sfac*c,4)}   '
      f'(agreement with fit: {abs(sfac-s):.2e})')
Cn = (Cl - c) * sfac

js = []
for i, k in enumerate(ks[::STRIDE]):
    j = ks.index(k)
    P = cm[im[k]['cid']]['params']
    js.append(dict(id=i, img_name=os.path.splitext(im[k]['name'])[0],
                   width=int(cm[im[k]['cid']]['w']), height=int(cm[im[k]['cid']]['h']),
                   position=[float(v) for v in Cn[j]],
                   rotation=[[float(v) for v in r] for r in (R @ qvec2R(im[k]['q']).T)],
                   fx=float(P[0]), fy=float(P[1])))
json.dump(js, open(OUT, 'w'))
print(f'wrote {OUT}: {len(js)} cameras (stride {STRIDE}, from {len(ks)})')
