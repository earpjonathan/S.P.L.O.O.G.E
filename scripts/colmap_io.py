import numpy as np, struct, os

def _r(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def read_cameras(p):
    cams = {}
    NP = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12,11:5}
    with open(p,'rb') as f:
        n = _r(f,'<Q')[0]
        for _ in range(n):
            cid, model, w, h = _r(f,'<iiQQ')
            npar = NP.get(model, 4)
            par = _r(f, '<'+'d'*npar)
            cams[cid] = dict(model=model, w=w, h=h, params=np.array(par))
    return cams

def read_images(p):
    imgs = {}
    with open(p,'rb') as f:
        n = _r(f,'<Q')[0]
        for _ in range(n):
            iid = _r(f,'<I')[0]
            q = np.array(_r(f,'<dddd'))       # w,x,y,z
            t = np.array(_r(f,'<ddd'))
            cid = _r(f,'<I')[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            npts = _r(f,'<Q')[0]
            d = np.frombuffer(f.read(24*npts), dtype=np.dtype([('x','<f8'),('y','<f8'),('id','<i8')]))
            imgs[iid] = dict(q=q, t=t, cid=cid, name=name.decode(),
                             xys=np.stack([d['x'],d['y']],1), p3d=d['id'].copy())
    return imgs

def read_points3D(p):
    ids=[];xyz=[];rgb=[];err=[];tl=[]
    with open(p,'rb') as f:
        n=_r(f,'<Q')[0]
        for _ in range(n):
            pid=_r(f,'<Q')[0]
            x,y,z=_r(f,'<ddd')
            r,g,b=_r(f,'<BBB')
            e=_r(f,'<d')[0]
            L=_r(f,'<Q')[0]
            f.read(8*L)
            ids.append(pid);xyz.append((x,y,z));rgb.append((r,g,b));err.append(e);tl.append(L)
    return (np.array(ids), np.array(xyz), np.array(rgb,dtype=np.uint8),
            np.array(err), np.array(tl))

def qvec2R(q):
    w,x,y,z=q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])

def centers(imgs):
    """world-space camera centers, sorted by image name"""
    ks = sorted(imgs, key=lambda k: imgs[k]['name'])
    C=[];R=[];names=[]
    for k in ks:
        Rm = qvec2R(imgs[k]['q'])
        C.append(-Rm.T @ imgs[k]['t']); R.append(Rm); names.append(imgs[k]['name'])
    return np.array(C), np.array(R), names
