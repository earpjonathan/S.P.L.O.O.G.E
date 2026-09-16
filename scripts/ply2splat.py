#!/usr/bin/env python
"""Fast vectorised 3DGS .ply -> .splat (antimatter15 format, 32 bytes/splat)."""
import numpy as np, sys, argparse

SH_C0 = 0.28209479177387814

def qmul(a,b):
    """quaternion product, both (...,4) in (w,x,y,z)"""
    w1,x1,y1,z1=a[...,0],a[...,1],a[...,2],a[...,3]
    w2,x2,y2,z2=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],-1)

def rot_between(a,b):
    """rotation matrix + quaternion taking unit vector a to unit vector b"""
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    v=np.cross(a,b); c=float(np.dot(a,b))
    if c < -0.999999:                       # antiparallel: pick any perpendicular
        axis=np.array([1.0,0,0]) if abs(a[0])<0.9 else np.array([0,1.0,0])
        v=np.cross(a,axis); v/=np.linalg.norm(v)
        q=np.array([0.0,v[0],v[1],v[2]])
    else:
        q=np.array([1+c,v[0],v[1],v[2]]); q/=np.linalg.norm(q)
    w,x,y,z=q
    R=np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
    return R,q

def read_ply(path):
    with open(path,'rb') as f:
        hdr=b''
        while b'end_header' not in hdr:
            c=f.read(1)
            if not c: raise RuntimeError('bad ply header')
            hdr+=c
        f.read(1)  # newline
        lines=hdr.decode('ascii',errors='replace').splitlines()
        if not any('binary_little_endian' in l for l in lines):
            raise RuntimeError('only binary_little_endian supported')
        vax=None
        for l in lines:
            if l.startswith('comment Vertical axis:'):
                vax=np.array([float(t) for t in l.split(':')[1].split()])
        props=[l.split()[-1] for l in lines if l.startswith('property')]
        types=[l.split()[1] for l in lines if l.startswith('property')]
        if any(t!='float' for t in types):
            raise RuntimeError(f'expected all float props, got {set(types)}')
        n=[int(l.split()[-1]) for l in lines if l.startswith('element vertex')][0]
        data=np.frombuffer(f.read(n*len(props)*4),dtype='<f4').reshape(n,len(props))
    return {p:data[:,i] for i,p in enumerate(props)}, n, vax

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('-o','--output',required=True)
    ap.add_argument('--drop-nan',action='store_true',default=True)
    ap.add_argument('--min-opacity',type=float,default=None,
                    help='drop splats with sigmoid(opacity) below this')
    ap.add_argument('--no-normalize',action='store_true',
                    help='keep raw COLMAP frame instead of centring/levelling/scaling')
    ap.add_argument('--up',default=None,
                    help='override the scene up direction as x,y,z in the PLY '
                         'frame. Brush comment Vertical axis is only an estimate '
                         'and can be wrong: on the 0026+0027 scene it left the '
                         'cameras below terrain.')
    ap.add_argument('--target-radius',type=float,default=3.0,
                    help='scale so the p90 splat radius equals this (default 3)')
    ap.add_argument('--colmap',default=None,
                    help='COLMAP sparse dir; normalise using CAMERA positions '
                         '(far better than splat median) and emit cameras.json')
    ap.add_argument('--cameras-out',default=None,help='where to write cameras.json')
    ap.add_argument('--cameras-stride',type=int,default=1,
                    help='keep every Nth camera in cameras.json')
    ap.add_argument('--strip-floaters',action='store_true',
                    help='drop sky/background blobs: big splats far from any '
                         'COLMAP-triangulated point. Requires --colmap.')
    ap.add_argument('--floater-agl',type=float,default=1.0,
                    help='drop splats more than this above local terrain '
                         '(scene units; 95%% of real splats sit within 0.09)')
    ap.add_argument('--floater-below',type=float,default=1.5,
                    help='drop splats more than this BELOW local terrain')
    ap.add_argument('--strip-far',action='store_true',
                    help='drop big splats far from any COLMAP-triangulated '
                         'point. Use on DENSELY TEXTURED scenes (city): real '
                         'surface there sits close to points, sky blobs do not. '
                         'Do NOT use on bare dirt/tarmac -- few features means '
                         'real ground looks far and this deletes terrain.')
    ap.add_argument('--far-dist',type=float,default=1.0,
                    help='distance from nearest COLMAP point to count as far')
    ap.add_argument('--far-size',type=float,default=0.05,
                    help='largest-axis scale above which a far splat is a blob')
    ap.add_argument('--sky-margin',type=float,default=2.0,
                    help='also drop anything higher than (highest camera + '
                         'this), everywhere including outside the flight area. '
                         'Nothing real is above the drone in an outdoor site.')
    ap.add_argument('--path',action='store_true',
                    help='bake the drone flight path into the splat as a '
                         'coloured track (cyan at takeoff -> magenta at the '
                         'end). Requires --colmap.')
    ap.add_argument('--path-radius',type=float,default=0.012,
                    help='thickness of the flight-path track (scene units)')
    ap.add_argument('--path-step',type=float,default=0.006,
                    help='spacing between path marker splats (scene units)')
    ap.add_argument('--max-splats',type=int,default=None,
                    help='cap output count, keeping the most visually significant '
                         '(opacity x projected size). Useful for web delivery: '
                         '.splat is 32 bytes/splat, so 1.5M ~= 48 MB')
    ap.add_argument('--ground-opacity-boost',type=float,default=None,metavar='K',
                    help='multiply GROUND optical depth by K via a -> 1-(1-a)^k. '
                         'The ground is a semi-transparent slab, not a surface, so '
                         'a low grazing ray saturates while a downward one exits '
                         '~60%% opaque -- this is the only post-hoc lever that '
                         'helps (every geometric fix made it worse). K=4 took the '
                         'worst frame from 54.7%% holes to 18.9%%. Off by default.')
    ap.add_argument('--ground-band',type=float,nargs=2,default=(-0.6,0.15),
                    metavar=('LO','HI'),
                    help='height band relative to the visible surface counted as '
                         'ground for --ground-opacity-boost (viewer units)')
    a=ap.parse_args()
    v,n,vax=read_ply(a.input)
    if a.up:
        vax=np.array([float(t) for t in a.up.split(',')])
        print(f'  up OVERRIDDEN to {np.round(vax/np.linalg.norm(vax),4)}')
    xyz=np.stack([v['x'],v['y'],v['z']],1)
    sc =np.exp(np.stack([v['scale_0'],v['scale_1'],v['scale_2']],1))
    rot=np.stack([v['rot_0'],v['rot_1'],v['rot_2'],v['rot_3']],1)
    op =1/(1+np.exp(-v['opacity']))
    rgb=0.5+SH_C0*np.stack([v['f_dc_0'],v['f_dc_1'],v['f_dc_2']],1)

    keep=np.ones(n,bool)
    bad=~(np.isfinite(xyz).all(1)&np.isfinite(sc).all(1)&np.isfinite(rot).all(1)&np.isfinite(op))
    if a.drop_nan and bad.any():
        keep&=~bad; print(f'  dropped {int(bad.sum())} non-finite splats')
    if a.min_opacity is not None:
        low=op<a.min_opacity
        keep&=~low; print(f'  dropped {int((low&~bad).sum())} splats below opacity {a.min_opacity}')
    xyz,sc,rot,op,rgb=xyz[keep],sc[keep],rot[keep],op[keep],rgb[keep]
    m=len(xyz)

    # --- optional COLMAP cameras, carried through the same transform ---
    cams=None
    if a.colmap:
        import sys as _s; _s.path.insert(0,'scripts')
        from colmap_io import read_images, read_cameras, qvec2R
        _im=read_images(a.colmap+'/images.bin'); _cm=read_cameras(a.colmap+'/cameras.bin')
        ks=sorted(_im, key=lambda k:_im[k]['name'])
        cams=[]
        for k in ks:
            Rw=qvec2R(_im[k]['q']); t=_im[k]['t']
            cams.append(dict(name=_im[k]['name'], C=(-Rw.T@t), R_c2w=Rw.T,
                             cam=_cm[_im[k]['cid']]))
        print(f'  loaded {len(cams)} COLMAP cameras from {a.colmap}')

    # --- normalise into the viewer's frame: level, centre, scale ---
    Rlev = np.eye(3); c = np.zeros(3); sfac = 1.0
    if not a.no_normalize:
        if vax is not None and np.isfinite(vax).all() and np.linalg.norm(vax) > 0:
            Rlev, qR = rot_between(np.asarray(vax, float), np.array([0.0, -1.0, 0.0]))
            xyz = xyz @ Rlev.T
            rot = qmul(np.broadcast_to(qR, rot.shape), rot)
            print(f'  levelled: vertical axis {np.round(vax,3)} -> (0,-1,0)')
        else:
            print('  no Vertical axis comment; skipping levelling')

        if cams:
            # Centre and scale on the CAMERA positions, not the splats: splat
            # medians are dragged around by sky/background blobs, which drops the
            # viewer's default camera inside the scene.
            Cs = np.array([x['C'] for x in cams]) @ Rlev.T
            c = Cs.mean(0)
            Cs = Cs - c
            r90 = np.percentile(np.linalg.norm(Cs, axis=1), 90)
            sfac = a.target_radius / r90 if r90 > 0 else 1.0
            xyz = (xyz - c) * sfac
            sc *= sfac
            Cs *= sfac
            for x, Cn in zip(cams, Cs):
                x['Cn'] = Cn
                x['Rn'] = Rlev @ x['R_c2w']
            print(f'  centred on CAMERA centroid {np.round(c,3)}, scaled x{sfac:.4f} '
                  f'(camera p90 radius {r90:.2f} -> {a.target_radius})')
        else:
            c = np.median(xyz, 0)
            xyz = xyz - c
            r90 = np.percentile(np.linalg.norm(xyz, axis=1), 90)
            sfac = a.target_radius / r90 if r90 > 0 else 1.0
            xyz *= sfac
            sc *= sfac
            print(f'  centred at {np.round(c,3)}, scaled x{sfac:.4f} '
                  f'(p90 radius {r90:.2f} -> {a.target_radius})')
    elif cams:
        for x in cams:
            x['Cn'] = x['C']; x['Rn'] = x['R_c2w']

    # --- strip far blobs, by distance to triangulated geometry ----------
    # The mirror image of --strip-floaters. On a textured scene (downtown)
    # distance-to-points is a good floater test; on bare ground it is not,
    # because it is really measuring texture. Needs no up direction.
    if a.strip_far:
        if not a.colmap:
            raise SystemExit('--strip-far needs --colmap')
        from scipy.spatial import cKDTree
        from colmap_io import read_points3D
        _, pxyz, _, _, _ = read_points3D(a.colmap + '/points3D.bin')
        P = pxyz @ Rlev.T
        if not a.no_normalize and cams:
            P = (P - c) * sfac
        dist, _ = cKDTree(P).query(xyz, k=1, workers=-1)
        big = np.sort(sc, axis=1)[:, 2]
        drop = (dist > a.far_dist) & (big > a.far_size)
        print(f'  far blobs: dropped {int(drop.sum()):,} of {m:,} '
              f'({100*drop.mean():.2f}%) with dist>{a.far_dist} and size>{a.far_size}')
        kept = ~drop
        xyz, sc, rot, op, rgb = xyz[kept], sc[kept], rot[kept], op[kept], rgb[kept]
        m = len(xyz)

    # --- strip sky floaters, by height above local terrain --------------
    # Sky has no parallax, so its depth is unconstrained: the optimiser can put
    # a sky-coloured blob anywhere along the ray and still match every image.
    # Many land low over the terrain and read as fake "clouds".
    #
    # Do NOT test distance-to-nearest-COLMAP-point for this. Textureless dirt
    # yields few SIFT features, so *real* ground splats there are far from any
    # triangulated point and, being smooth, large -- that rule deleted 14.6 %
    # real terrain and punched black holes in the ground.
    #
    # Height above a local terrain field separates them properly: real surface
    # splats sit at terrain height whatever their texture, floaters do not.
    # Runs BEFORE --max-splats so the budget goes on real geometry.
    if a.strip_floaters:
        if not a.colmap:
            raise SystemExit('--strip-floaters needs --colmap (the '
                             'triangulated points define the terrain)')
        from colmap_io import read_points3D
        from agl import terrain_field, agl
        _, pxyz, _, _, _ = read_points3D(a.colmap + '/points3D.bin')
        P = pxyz @ Rlev.T
        if not a.no_normalize and cams:
            P = (P - c) * sfac
        Cn = np.array([x['Cn'] for x in cams])
        # NOTE: Brush's "Vertical axis" comment points DOWN (gravity), and
        # ply2splat maps it to (0,-1,0), so after levelling UP IS +Y. Checked
        # two ways: 66 % of COLMAP points lie below the cameras, and 78 % of
        # frames have forward.Y < 0 while the drone plainly looks down.
        ch = Cn[:, [0, 2]]
        glo, ghi = ch.min(0), ch.max(0)
        pad = 0.6 * (ghi - glo); glo -= pad; ghi += pad
        grid, tlo, thi = terrain_field(P[:, [0, 2]], P[:, 1], res=96, pct=60,
                                       extent=(glo, ghi))
        A = agl(xyz[:, [0, 2]], xyz[:, 1], grid, tlo, thi)
        inside = ((xyz[:, [0, 2]] >= glo) & (xyz[:, [0, 2]] <= ghi)).all(1)
        # Outside the flight area is distant backdrop (city, hills) with no
        # terrain estimate -- leave it alone rather than guess.
        drop = inside & ((A > a.floater_agl) | (A < -a.floater_below))
        # Outside the flight area there is no terrain estimate, but a global
        # ceiling still applies: sky splats there reach height 139 while the
        # highest camera is 4.4. Nothing real is above the drone.
        if a.sky_margin is not None:
            ceil = Cn[:, 1].max() + a.sky_margin
            drop |= xyz[:, 1] > ceil
        kept = ~drop
        print(f'  floaters: dropped {int(drop.sum()):,} of {m:,} '
              f'({100*drop.mean():.2f}%) at agl>{a.floater_agl} or '
              f'agl<-{a.floater_below} inside the flight area '
              f'(drone reached agl {agl(ch, Cn[:,1], grid, tlo, thi).max():.2f}), '
              f'sky ceiling {Cn[:,1].max() + a.sky_margin:.2f}')
        xyz, sc, rot, op, rgb = xyz[kept], sc[kept], rot[kept], op[kept], rgb[kept]
        m = len(xyz)

    # rank by visual significance: opacity x projected area (2 largest axes)
    ss=np.sort(sc,axis=1)
    signif=op*ss[:,1]*ss[:,2]
    if a.max_splats is not None and m>a.max_splats:
        sel=np.argpartition(-signif,a.max_splats)[:a.max_splats]
        xyz,sc,rot,op,rgb,signif=xyz[sel],sc[sel],rot[sel],op[sel],rgb[sel],signif[sel]
        print(f'  capped {m:,} -> {a.max_splats:,} splats by opacity x projected area')
        m=len(xyz)

    # biggest / most opaque first (viewer streams progressively)
    order=np.argsort(-signif)
    xyz,sc,rot,op,rgb=xyz[order],sc[order],rot[order],op[order],rgb[order]

    # --- ground opacity boost -------------------------------------------
    # Applied AFTER the cap on purpose: boosting first would inflate ground
    # splats' significance and evict non-ground ones, changing WHICH splats
    # survive. After the cap, the same splats are kept as an unboosted build
    # and only the ground gets denser, so builds stay comparable.
    if a.ground_opacity_boost is not None and a.ground_opacity_boost != 1.0:
        import sys as _s2; _s2.path.insert(0,'scripts')
        from flatten_ground import visible_surface
        K = float(a.ground_opacity_boost)
        RES = 320
        glo = np.percentile(xyz[:,[0,2]],1,axis=0)
        ghi = np.percentile(xyz[:,[0,2]],99,axis=0)
        surf,_gm = visible_surface(xyz, op, sc, RES, glo, ghi)
        gix = np.clip(((xyz[:,[0,2]]-glo)/(ghi-glo)*RES).astype(int),0,RES-1)
        hh = xyz[:,1] - surf[gix[:,0],gix[:,1]]
        band = (hh > a.ground_band[0]) & (hh < a.ground_band[1])
        before = op[band].mean() if band.any() else 0.0
        # a -> 1-(1-a)^K multiplies optical depth tau = -ln(1-a) by exactly K,
        # in every direction, which is why it helps a slab whose problem is
        # path length rather than geometry.
        op[band] = 1.0 - np.power(1.0 - op[band], K)
        print(f'  ground opacity boost K={K}: {int(band.sum()):,} splats '
              f'({100*band.mean():.1f}%) in band {a.ground_band[0]}..{a.ground_band[1]}, '
              f'mean opacity {before:.3f} -> {op[band].mean():.3f}')

    # --- optional flight path, baked in as splats -----------------------
    # Drawn as Gaussians rather than a GL line primitive on purpose: this
    # viewer has no depth buffer, it sorts and alpha-blends. A line would
    # draw straight through the terrain; splats join the same sort and are
    # occluded correctly. Appended AFTER the cap so they are never culled.
    if a.path:
        if not cams:
            raise SystemExit('--path needs --colmap (it draws the camera track)')
        C = np.array([x['Cn'] for x in cams])          # already in viewer frame
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        nsub = np.maximum(1, np.ceil(seg / a.path_step).astype(int))
        pts, tt = [], []
        for i, k in enumerate(nsub):
            f = np.arange(k)[:, None] / k
            pts.append(C[i] * (1 - f) + C[i + 1] * f)
            tt.append((i + f[:, 0]) / len(nsub))
        pts = np.concatenate(pts); tt = np.concatenate(tt)
        # cyan -> magenta over the flight, so direction of travel is readable
        # against brown/green terrain
        pc = np.stack([tt, 1 - tt * 0.85, np.ones_like(tt)], 1)
        k = len(pts)
        xyz = np.concatenate([xyz, pts])
        sc  = np.concatenate([sc, np.full((k, 3), a.path_radius, np.float32)])
        rot = np.concatenate([rot, np.tile([1.0, 0, 0, 0], (k, 1))])
        op  = np.concatenate([op, np.ones(k, np.float32)])
        rgb = np.concatenate([rgb, pc])
        m += k
        print(f'  flight path: {k:,} marker splats along {len(cams)} poses '
              f'(radius {a.path_radius}, step {a.path_step})')

    nrm=np.linalg.norm(rot,axis=1,keepdims=True); nrm[nrm==0]=1
    rotb=np.clip(rot/nrm*128+128,0,255).astype(np.uint8)
    rgba=np.empty((m,4),np.uint8)
    rgba[:,:3]=np.clip(rgb*255,0,255).astype(np.uint8)
    rgba[:,3]=np.clip(op*255,0,255).astype(np.uint8)

    buf=np.empty((m,32),np.uint8)
    buf[:,0:12]=xyz.astype('<f4').view(np.uint8).reshape(m,12)
    buf[:,12:24]=sc.astype('<f4').view(np.uint8).reshape(m,12)
    buf[:,24:28]=rgba
    buf[:,28:32]=rotb
    buf.tofile(a.output)
    print(f'  {a.input} -> {a.output}: {m:,} splats, {m*32/1048576:.1f} MB')

    if cams and (a.cameras_out or a.colmap):
        import json, os
        out = a.cameras_out or os.path.join(os.path.dirname(a.output) or '.', 'cameras.json')
        js = []
        for i, x in enumerate(cams[::a.cameras_stride]):
            cm = x['cam']; W, H = int(cm['w']), int(cm['h']); P = cm['params']
            js.append(dict(id=i, img_name=os.path.splitext(x['name'])[0],
                           width=W, height=H,
                           position=[float(v) for v in x['Cn']],
                           rotation=[[float(v) for v in r] for r in x['Rn']],
                           fx=float(P[0]), fy=float(P[1])))
        with open(out, 'w') as f:
            json.dump(js, f)
        print(f'  wrote {out}: {len(js)} cameras '
              f'(stride {a.cameras_stride}, from {len(cams)})')

if __name__ == '__main__':
    main()
