import json, numpy as np, cv2, glob, sys
CLIP=sys.argv[1]; T0=float(sys.argv[2]); DIR=sys.argv[3]; W_FULL=int(sys.argv[4])
files=sorted(glob.glob(DIR+'/*.jpg'))
d=json.load(open(f'gyro/cam_{CLIP}.json'))
q=np.array([e['org_quat'] for e in d]); ts=np.array([e['timestamp_ms'] for e in d])
start=int(np.argmin(np.abs(ts-T0*1000)))
img0=cv2.imread(files[0],cv2.IMREAD_GRAYSCALE); H,W=img0.shape
cx,cy=W/2,H/2; scale=W_FULL/W
def qrel(a,b):
    w1,x1,y1,z1=a; w1,x1,y1,z1=w1,-x1,-y1,-z1
    w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
lk=dict(winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,0.01))
R=[];Dsp=[];Ang=[]
prev=img0
for i in range(1,len(files)):
    cur=cv2.imread(files[i],cv2.IMREAD_GRAYSCALE)
    p0=cv2.goodFeaturesToTrack(prev,maxCorners=2500,qualityLevel=0.005,minDistance=10,blockSize=7)
    if p0 is not None and len(p0)>80:
        p1,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p0,None,**lk)
        p0r,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,p1,None,**lk)
        g=(st.ravel()==1)&(st2.ravel()==1)
        g&=np.linalg.norm((p0r-p0).reshape(-1,2),axis=1)<1.0
        qr=qrel(q[start+i-1],q[start+i])
        ang=2*np.arccos(np.clip(abs(qr[0]),-1,1))
        if g.sum()>60 and ang>np.radians(0.5):
            a=p0.reshape(-1,2)[g];b=p1.reshape(-1,2)[g]
            rr=np.hypot(a[:,0]-cx,a[:,1]-cy)
            dd=np.linalg.norm(b-a,axis=1)
            R.append(rr);Dsp.append(dd);Ang.append(np.full(len(rr),ang))
    prev=cur
R=np.concatenate(R);Dsp=np.concatenate(Dsp);Ang=np.concatenate(Ang)
slope=Dsp/Ang
print(f'{len(R):,} feature observations')
print()
print(f'{"r (burst px)":>13}{"r/halfW":>9}{"n":>8}{"slope px/rad":>14}{"-> @full-res":>13}')
edges=np.linspace(0,np.hypot(cx,cy),11)
prof=[]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(R>=lo)&(R<hi)
    if m.sum()<200: continue
    s=np.median(slope[m])
    prof.append((0.5*(lo+hi),s,m.sum()))
    print(f'{0.5*(lo+hi):>13.0f}{0.5*(lo+hi)/(W/2):>9.2f}{m.sum():>8}{s:>14.0f}{s*scale:>13.0f}')
prof=np.array(prof)
r0,s0=prof[0,0],prof[0,1]
print()
print(f'slope at centre {prof[0,1]*scale:.0f} px/rad -> edge {prof[-1,1]*scale:.0f} px/rad   ratio {prof[-1,1]/prof[0,1]:.2f}')
f=prof[0,1]*scale
print(f'PINHOLE would predict ratio 1+(r/f)^2 = {1+(prof[-1,0]*scale/f)**2:.2f}')
print(f'EQUIDISTANT fisheye would predict ratio ~1.00')
