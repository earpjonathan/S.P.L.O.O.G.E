#!/usr/bin/env python
"""COLMAP poses -> time-resampled flight trajectories in the VIEWER's frame.

The viewer frame is whatever ply2splat.py produced: levelled by Rlev, centred
on the camera centroid, scaled by sfac. Rather than re-deriving that (which
would mean knowing exactly which --up was passed on the day), this recovers
the transform from an existing cameras.json, which is already in the viewer
frame:

    Rn   = Rlev @ R_c2w          ->  Rlev = Rn @ R_c2w.T   (exact, from one cam)
    Cn   = (C @ Rlev.T - c)*s    ->  least squares for s and (-s*c)

Then every COLMAP pose goes through the same transform, so the ghosts land on
the splats they were reconstructed from.

Time comes from the frame index in the filename (CLIP_FRAMEIDX.jpg) over the
capture fps -- NOT from the pose index. Frame selection was adaptive (constant
inter-frame rotation), so poses are dense in the turns and sparse on the
straights. Playing them back at a fixed rate per pose would make the drone
crawl through corners and teleport down straights; playing them on real time
is what makes the replay a replay.
"""
import numpy as np, json, os, argparse, re
from scipy.spatial.transform import Rotation, Slerp

NAME_RE = re.compile(r'^(\d+)_(\d+)$')


def recover_transform(cams_json, colmap_dir):
    """Solve Rlev, c, sfac by matching cameras.json against the COLMAP model."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from colmap_io import read_images, qvec2R

    im = read_images(os.path.join(colmap_dir, 'images.bin'))
    by_name = {}
    for k in im:
        Rw = qvec2R(im[k]['q']); t = im[k]['t']
        by_name[os.path.splitext(im[k]['name'])[0]] = (-Rw.T @ t, Rw.T)

    js = json.load(open(cams_json))
    matched = [(np.array(e['position'], float), np.array(e['rotation'], float),
                *by_name[e['img_name']])
               for e in js if e['img_name'] in by_name]
    if len(matched) < 2:
        raise SystemExit(f'only {len(matched)} of {len(js)} cameras.json entries '
                         f'matched {colmap_dir} -- wrong model?')

    # Rlev from the rotations: average, then re-orthogonalise via SVD.
    M = np.zeros((3, 3))
    for Cn, Rn, C, Rc2w in matched:
        M += Rn @ Rc2w.T
    U, _, Vt = np.linalg.svd(M / len(matched))
    Rlev = U @ Vt
    if np.linalg.det(Rlev) < 0:
        U[:, -1] *= -1
        Rlev = U @ Vt

    # Cn = s*Cl + b  (b = -s*c). Stack all matched cameras, 3 rows each.
    Cl = np.array([C for _, _, C, _ in matched]) @ Rlev.T
    Cn = np.array([c for c, _, _, _ in matched])
    A = np.zeros((3 * len(matched), 4))
    y = Cn.reshape(-1)
    A[:, 0] = Cl.reshape(-1)
    for i in range(3):
        A[i::3, 1 + i] = 1.0
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    sfac, b = sol[0], sol[1:]
    resid = np.abs(A @ sol - y).max()

    rot_resid = max(np.abs(Rn - Rlev @ Rc2w).max() for _, Rn, _, Rc2w in matched)
    print(f'  transform recovered from {len(matched)} cameras: '
          f'scale {sfac:.5f}, max position residual {resid:.2e}, '
          f'max rotation residual {rot_resid:.2e}')
    if resid > 1e-3 or rot_resid > 1e-3:
        raise SystemExit('  residual too large -- cameras.json and the COLMAP '
                         'model disagree. Is this the sparse dir it was built from?')
    return Rlev, sfac, b, by_name


def estimate_tilt(P, Rn, level=0.25):
    """FPV cameras are bolted on at an upward tilt, so camera forward is not
    body forward: render the drone in the camera frame and it sits permanently
    nose-up by the mount angle.

    Estimate the mount angle as the median angle, in the camera's own vertical
    plane, between camera forward (+Z) and the direction of travel.

    Two filters, both load-bearing. Direction of travel is meaningless at a
    hover, so only samples above the median speed count. And it is only equal
    to body forward in roughly LEVEL flight -- climb vertically while looking
    out at a building and the angle reads ~90 deg and poisons the median. That
    is exactly what happened on the downtown clips, which are slow and full of
    verticals: they disagreed 49/26/16 deg while the two site flights agreed to
    0.8. `level` keeps only samples whose travel is within asin(level) of
    horizontal.

    Returns (radians, n_used, iqr_deg). A wide IQR means the flight did not
    constrain it and the caller should not trust the number."""
    v = np.gradient(P, axis=0)
    sp = np.linalg.norm(v, axis=1)
    vn = v / np.maximum(sp, 1e-12)[:, None]
    up_w = -Rn[:, :, 1]                       # camera -Y in world
    fwd = Rn[:, :, 2]                         # camera +Z in world
    # world up is +Y after levelling (see ply2splat: Vertical axis -> (0,-1,0))
    climb = np.abs(vn[:, 1])
    ok = (sp > np.percentile(sp, 50)) & (climb < level)
    if ok.sum() < 20:
        return 0.0, 0, float('inf')
    ang = np.arctan2(np.einsum('ij,ij->i', vn[ok], up_w[ok]),
                     np.einsum('ij,ij->i', vn[ok], fwd[ok]))
    q1, q3 = np.percentile(ang, [25, 75])
    return float(np.median(ang)), int(ok.sum()), float(np.rad2deg(q3 - q1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--colmap', required=True, help='COLMAP sparse dir')
    ap.add_argument('--cameras', required=True, help='cameras.json in viewer frame')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--fps', type=float, default=50.0, help='capture fps')
    ap.add_argument('--hz', type=float, default=30.0, help='resample rate')
    ap.add_argument('--gap', type=float, default=1.0,
                    help='seconds between registered poses above which the '
                         'segment is marked interpolated (drawn dashed)')
    ap.add_argument('--tilt', type=float, default=None,
                    help='camera up-tilt in degrees; default estimates it from '
                         'the flight')
    ap.add_argument('--tilt-iqr', type=float, default=25.0,
                    help='reject an estimated tilt whose interquartile spread '
                         'exceeds this many degrees (the flight did not '
                         'constrain it)')
    ap.add_argument('--no-tilt', action='store_true',
                    help='render the camera frame itself, not a levelled body')
    a = ap.parse_args()

    Rlev, sfac, b, by_name = recover_transform(a.cameras, a.colmap)

    # every registered pose, grouped by clip
    clips = {}
    for name, (C, Rc2w) in by_name.items():
        m = NAME_RE.match(name)
        if not m:
            print(f'  skipping unparseable name {name}')
            continue
        clip, idx = m.group(1), int(m.group(2))
        clips.setdefault(clip, []).append((idx, C, Rc2w))

    out = {'hz': a.hz, 'clips': []}
    for clip in sorted(clips):
        rows = sorted(clips[clip])
        idx = np.array([r[0] for r in rows])
        C = np.array([r[1] for r in rows])
        Rc2w = np.array([r[2] for r in rows])

        t = idx / a.fps
        P = (C @ Rlev.T) * sfac + b
        Rn = np.einsum('ij,njk->nik', Rlev, Rc2w)

        # --- resample onto a uniform real-time clock ---
        tt = np.arange(t[0], t[-1], 1.0 / a.hz)
        Pr = np.stack([np.interp(tt, t, P[:, i]) for i in range(3)], 1)
        rots = Rotation.from_matrix(Rn)
        Rr = Slerp(t, rots)(tt)
        qr = Rr.as_quat()                        # (x,y,z,w)

        # gaps: where the nearest registered pose is far away in time
        j = np.searchsorted(t, tt).clip(1, len(t) - 1)
        gap_len = t[j] - t[j - 1]
        interp = gap_len > a.gap

        dt = 1.0 / a.hz
        vel = np.gradient(Pr, dt, axis=0)
        speed = np.linalg.norm(vel, axis=1)

        tilt, nused, iqr = estimate_tilt(Pr, Rr.as_matrix())
        note = f' from {nused} level samples, IQR {iqr:.0f} deg'
        if iqr > a.tilt_iqr and a.tilt is None:
            note += f' -- IQR>{a.tilt_iqr:.0f}, NOT TRUSTED, falling back to 0'
            tilt = 0.0
        if a.tilt is not None:
            note += f' -- overridden to {a.tilt:+.1f}'
            tilt = np.deg2rad(a.tilt)
        if a.no_tilt:
            tilt, note = 0.0, ' (--no-tilt: raw camera frame)'
        # rotate about the camera's own X axis to bring +Z onto the flight
        # direction; body = cam * Rx(tilt)
        qbody = (Rr * Rotation.from_rotvec([tilt, 0, 0])).as_quat() \
            if tilt else qr

        out['clips'].append(dict(
            name=clip,
            n=len(tt),
            t0=float(tt[0]),
            dt=dt,
            duration=float(tt[-1] - tt[0]),
            poses=len(t),
            tilt_deg=float(np.rad2deg(tilt)),
            pos=[round(float(x), 4) for x in Pr.reshape(-1)],
            quat=[round(float(x), 5) for x in qbody.reshape(-1)],
            cam_quat=[round(float(x), 5) for x in qr.reshape(-1)],
            speed=[round(float(x), 4) for x in speed],
            interp=[int(x) for x in interp],
        ))
        print(f'  {clip}: {len(t)} poses over {t[-1]-t[0]:6.1f} s '
              f'({len(t)/(t[-1]-t[0]):4.1f} Hz mean) -> {len(tt)} samples, '
              f'gaps>{a.gap}s: {100*interp.mean():4.1f}%, '
              f'speed p50 {np.percentile(speed,50):.2f} p99 {np.percentile(speed,99):.2f} u/s, '
              f'tilt {np.rad2deg(tilt):+.1f} deg' + note)

    with open(a.output, 'w') as f:
        json.dump(out, f)
    print(f'  wrote {a.output} ({os.path.getsize(a.output)/1048576:.2f} MB, '
          f'{len(out["clips"])} clips)')


if __name__ == '__main__':
    main()
