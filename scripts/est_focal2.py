import json, numpy as np, cv2, glob, sys
CLIP=sys.argv[1]; T0=float(sys.argv[2]); DIR=sys.argv[3]
W_FULL,H_FULL=int(sys.argv[4]),int(sys.argv[5])
files=sorted(glob.glob(DIR+'/*.jpg'))
d=json.load(open(f'gyro/cam_{CLIP}.json'))
q=np.array([e['org_quat'] for e in d]); ts=np.array([e['timestamp_ms'] for e in d])
start=int(np.argmin(np.abs(ts-T0*1000)))
img0=cv2.imread(files[0],cv2.IMREAD_GRAYSCALE); H,W=img0.shape
scale=W_FULL/W
print(f'{CLIP}: burst start frame {start} (ts={ts[start]:.0f}ms), {len(files)} frames, {W}x{H}, scale={scale}')
def qrel(a,b):
    w1,x1,y1,z1=a; w1,x1,y1,z1=w1,-x1,-y1,-z1
    w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
lk=dict(winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,0.01))
rows=[];prev=img0
for i in range(1,len(files)):
    cur=cv2.imread(files[i],cv2.IMREAD_GRAYSCALE)
    p0=cv2.goodFeaturesToTrack(prev,maxCorners=1500,qualityLevel=0.01,minDistance=12,blockSize=7)
    if p0 is not None and len(p0)>60:
        p1,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p0,None,**lk)
        p0r,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,p1,None,**lk)
        g=(st.ravel()==1)&(st2.ravel()==1)
        g&=np.linalg.norm((p0r-p0).reshape(-1,2),axis=1)<1.0
        if g.sum()>50:
            a=p0.reshape(-1,2)[g];b=p1.reshape(-1,2)[g]
            med=np.median(np.linalg.norm(b-a,axis=1))
            qr=qrel(q[start+i-1],q[start+i])
            ang=2*np.arccos(np.clip(abs(qr[0]),-1,1))
            rows.append((ang,med))
    prev=cur
r=np.array(rows); print(f'usable pairs {len(r)}')
A,F=r[:,0],r[:,1]
m=A>np.radians(0.3); A,F=A[m],F[m]
f=np.sum(A*F)/np.sum(A*A)
for _ in range(8):
    res=F-f*A; s=1.4826*np.median(np.abs(res-np.median(res)))
    k=np.abs(res)<2.5*max(s,1e-9)
    f=np.sum(A[k]*F[k])/np.sum(A[k]*A[k])
r2=1-np.sum((F-f*A)**2)/np.sum((F-F.mean())**2)
ff=f*scale
print(f'  focal(burst)={f:.1f}  R2={r2:.3f}  inliers {k.sum()}/{len(A)}')
print(f'  FOCAL @{W_FULL} = {ff:.1f}   f/W={ff/W_FULL:.4f}')
print(f'  HFOV {2*np.degrees(np.arctan(W_FULL/2/ff)):.1f}  VFOV {2*np.degrees(np.arctan(H_FULL/2/ff)):.1f}  DFOV {2*np.degrees(np.arctan(np.hypot(W_FULL,H_FULL)/2/ff)):.1f}')
