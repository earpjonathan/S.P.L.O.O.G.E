import json, numpy as np, cv2, glob, os
W_FULL, H_FULL = 2688, 2016
files = sorted(glob.glob('work/burst/*.jpg'))
d = json.load(open('gyro/camera_0049.json'))
q = np.array([e['org_quat'] for e in d])
ts = np.array([e['timestamp_ms'] for e in d])

# burst begins at video t=258s -> nearest frame index
start = int(np.argmin(np.abs(ts - 258000.0)))
print(f'burst start frame index {start} (ts={ts[start]:.1f} ms), {len(files)} frames')

img0 = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
H, W = img0.shape
scale = W_FULL / W
print(f'burst res {W}x{H}, scale to full = {scale}')

def qrel(a, b):           # conj(a)*b
    w1,x1,y1,z1 = a; w1,x1,y1,z1 = w1,-x1,-y1,-z1
    w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

rows = []
prev = img0
lk = dict(winSize=(21,21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
for i in range(1, len(files)):
    cur = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=1200, qualityLevel=0.01,
                                 minDistance=12, blockSize=7)
    if p0 is not None and len(p0) > 60:
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
        good = (st.ravel()==1) & (st2.ravel()==1)
        fb = np.linalg.norm((p0r-p0).reshape(-1,2), axis=1)
        good &= fb < 1.0
        if good.sum() > 50:
            a = p0.reshape(-1,2)[good]; b = p1.reshape(-1,2)[good]
            fl = np.linalg.norm(b-a, axis=1)
            med = np.median(fl)
            qr = qrel(q[start+i-1], q[start+i])
            ang = 2*np.arccos(np.clip(abs(qr[0]),-1,1))     # rad
            rows.append((ang, med, good.sum(), np.median(np.abs(b[:,0]-a[:,0])), np.median(np.abs(b[:,1]-a[:,1]))))
    prev = cur

r = np.array(rows)
print(f'usable pairs: {len(r)}')
ang, flow = r[:,0], r[:,1]
m = ang > np.radians(0.15)          # only pairs with real rotation
print(f'pairs with rotation >0.15 deg: {m.sum()}')
A, F = ang[m], flow[m]
# robust slope through origin, iterated
f_half = np.sum(A*F)/np.sum(A*A)
for _ in range(6):
    res = F - f_half*A
    s = 1.4826*np.median(np.abs(res-np.median(res)))
    keep = np.abs(res) < 2.5*max(s,1e-9)
    f_half = np.sum(A[keep]*F[keep])/np.sum(A[keep]*A[keep])
pred = f_half*A
ss = 1 - np.sum((F-pred)**2)/np.sum((F-F.mean())**2)
print()
print(f'  focal (burst px)  = {f_half:.1f}')
print(f'  R^2               = {ss:.3f}   (inliers {keep.sum()}/{m.sum()})')
f_full = f_half*scale
print(f'  FOCAL (full-res px @ {W_FULL}) = {f_full:.1f}')
hfov = 2*np.degrees(np.arctan(W_FULL/2/f_full))
vfov = 2*np.degrees(np.arctan(H_FULL/2/f_full))
dfov = 2*np.degrees(np.arctan(np.hypot(W_FULL,H_FULL)/2/f_full))
print(f'  => HFOV {hfov:.1f} deg,  VFOV {vfov:.1f} deg,  DFOV {dfov:.1f} deg')
print(f'  => f/W ratio = {f_full/W_FULL:.4f}')
np.save('work/focal_rows.npy', r)
