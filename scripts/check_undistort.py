#!/usr/bin/env python
"""Does the undistorted proxy put features where the RENDER puts the geometry?

Two independent checks, because either alone can pass while the overlay is
still wrong:

A. MATH -- for a COLMAP-registered frame, take every observed 2D feature that
   has a triangulated 3D point. Project that 3D point through the plain pinhole
   camera (fx, fy, cx, cy) -- which is exactly what the splat viewer draws --
   and measure how far the OBSERVED feature is from it. That distance is the
   overlay error of the raw footage. Then undistort the observation and measure
   again.

B. PIXELS -- SIFT-match the raw proxy frame against the undistorted proxy frame
   at the same timestamp and check the actual displacement agrees with the map.
   This is what catches a wrong scale factor, a wrong principal point, or the
   video being a different frame than we think.
"""
import sys, subprocess, tempfile, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from colmap_io import read_images, read_points3D, qvec2R

MODEL = 'colmap/both/sparse/0'
CLIP, FPS, W, H = '0026', 50.0, 640, 480
FX, FY, CX, CY = 725.60519, 722.77797, 960.0, 720.0
D = np.array([-0.10878, 0.00815, 0.00077, 0.00037])
S = W / 1920.0

imgs = read_images(MODEL + '/images.bin')
ids, xyz, _, _, _ = read_points3D(MODEL + '/points3D.bin')
P3 = dict(zip(ids.tolist(), xyz))

cand = [v for v in imgs.values() if v['name'].startswith(CLIP)]
cand.sort(key=lambda v: (v['p3d'] >= 0).sum())
im = cand[-1]                                   # most observations
idx = int(im['name'].split('_')[1].split('.')[0])
t = idx / FPS
m = im['p3d'] >= 0
obs = im['xys'][m]
pts = np.array([P3[i] for i in im['p3d'][m]])
R, tv = qvec2R(im['q']), im['t']
X = pts @ R.T + tv
ok = X[:, 2] > 0.1
X, obs = X[ok], obs[ok]

pin = np.stack([FX * X[:, 0] / X[:, 2] + CX, FY * X[:, 1] / X[:, 2] + CY], 1)
und = cv2.undistortPoints(obs.reshape(-1, 1, 2).astype(np.float64),
                          np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.]]), D,
                          P=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.]])
                          ).reshape(-1, 2)
r_raw = np.linalg.norm(obs - pin, axis=1) * S       # in proxy pixels
r_und = np.linalg.norm(und - pin, axis=1) * S
rad = np.linalg.norm(obs - [CX, CY], axis=1) / np.hypot(CX, CY)

print(f'A. MATH  frame {im["name"]}  t={t:.2f}s  {len(obs)} points with 3D')
print(f'   raw footage vs render   median {np.median(r_raw):6.2f} px   '
      f'p90 {np.percentile(r_raw,90):6.2f}   max {r_raw.max():6.2f}   (of {W} wide)')
print(f'   undistorted vs render   median {np.median(r_und):6.2f} px   '
      f'p90 {np.percentile(r_und,90):6.2f}   max {r_und.max():6.2f}')
for lo, hi in [(0, .33), (.33, .66), (.66, 1.1)]:
    s = (rad >= lo) & (rad < hi)
    if s.sum():
        print(f'     r={lo:.2f}-{hi:.2f} ({s.sum():5d} pts)  '
              f'raw {np.median(r_raw[s]):6.2f} -> und {np.median(r_und[s]):5.2f}')

# ---------- B ----------
tmp = tempfile.mkdtemp()
for tag, src in (('u', 'viewer/video/%s.mp4' % CLIP),
                 ('r', 'viewer/video/%s_raw.mp4' % CLIP)):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-ss', f'{t:.4f}', '-i', src, '-frames:v', '1',
                    f'{tmp}/{tag}.png'], check=True)
A = cv2.imread(f'{tmp}/r.png', 0)
B = cv2.imread(f'{tmp}/u.png', 0)
sift = cv2.SIFT_create(4000)
ka, da = sift.detectAndCompute(A, None)
kb, db = sift.detectAndCompute(B, None)
mt = cv2.BFMatcher().knnMatch(da, db, k=2)
good = [p for p, q in mt if p.distance < 0.75 * q.distance]
pa = np.float32([ka[g.queryIdx].pt for g in good])
pb = np.float32([kb[g.trainIdx].pt for g in good])
# where SHOULD a raw-proxy point appear in the undistorted proxy?
Kp = np.array([[FX * S, 0, CX * S - .5], [0, FY * S, CY * S - .5], [0, 0, 1.]])
pred = cv2.undistortPoints(pa.reshape(-1, 1, 2), Kp, D, P=Kp).reshape(-1, 2)
err = np.linalg.norm(pred - pb, axis=1)
keep = err < 5                       # drop SIFT mismatches, then report
print(f'\nB. PIXELS  {len(good)} SIFT matches raw<->undistorted, '
      f'{keep.sum()} within 5 px of prediction')
print(f'   residual  median {np.median(err[keep]):.3f} px   '
      f'p90 {np.percentile(err[keep],90):.3f}   max {err[keep].max():.3f}')
print(f'   displacement applied: median {np.median(np.linalg.norm(pa-pb,axis=1)):.1f} px, '
      f'max {np.linalg.norm(pa-pb,axis=1).max():.1f} px')
