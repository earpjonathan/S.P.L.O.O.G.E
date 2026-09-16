import numpy as np, cv2, glob
SCALE = 2688/1344.0
lk = dict(winSize=(21,21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
print(f'{"t(s)":>5} {"medFlow50fps":>13} {"p90":>7} {"feat":>6}   flow in FULL-RES px/frame')
allmed=[]
for t in [50,100,150,200,230]:
    files=sorted(glob.glob(f'work/b{t}/*.jpg'))
    meds=[];feats=[]
    prev=cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)
    for f in files[1:]:
        cur=cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        p0=cv2.goodFeaturesToTrack(prev,maxCorners=1500,qualityLevel=0.01,minDistance=10,blockSize=7)
        if p0 is not None and len(p0)>60:
            p1,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p0,None,**lk)
            p0r,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,p1,None,**lk)
            g=(st.ravel()==1)&(st2.ravel()==1)
            g&=np.linalg.norm((p0r-p0).reshape(-1,2),axis=1)<1.0
            if g.sum()>40:
                a=p0.reshape(-1,2)[g];b=p1.reshape(-1,2)[g]
                meds.append(np.median(np.linalg.norm(b-a,axis=1))*SCALE)
                feats.append(g.sum())
        prev=cur
    meds=np.array(meds)
    allmed.append(meds)
    print(f'{t:>5} {np.median(meds):>13.1f} {np.percentile(meds,90):>7.1f} {int(np.median(feats)):>6}')
m=np.concatenate(allmed)
print()
print(f'ALL BURSTS: median {np.median(m):.1f} px/frame  p25 {np.percentile(m,25):.1f}  p75 {np.percentile(m,75):.1f}  p95 {np.percentile(m,95):.1f}')
print()
print('=== implied displacement if we decimate (full-res px between kept frames) ===')
print(f'{"keep every":>11} {"eff fps":>8} {"median disp":>12} {"p95 disp":>10} {"% of width":>11}')
for n in [1,2,3,4,5,6,8,10]:
    print(f'{n:>11} {50/n:>8.1f} {np.median(m)*n:>12.1f} {np.percentile(m,95)*n:>10.1f} {np.median(m)*n/2688*100:>10.1f}%')
