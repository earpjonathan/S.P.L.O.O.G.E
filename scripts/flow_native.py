import numpy as np, cv2, glob
S=2.0
lk=dict(winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
print(f'{"t(s)":>5} {"med@50fps":>10} {"p95":>7} {"med@10fps(x5)":>14} {"feat":>6} {"sharp":>7}')
for t in [5,12,20,28,35,50,100,150,200,230]:
    d=f'work/n{t}' if t in (5,12,20,28,35) else f'work/b{t}'
    files=sorted(glob.glob(d+'/*.jpg'))
    if not files: continue
    meds=[];feats=[];shs=[]
    prev=cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)
    for f in files[1:]:
        cur=cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        shs.append(cv2.Laplacian(cur,cv2.CV_64F).var())
        p0=cv2.goodFeaturesToTrack(prev,maxCorners=1500,qualityLevel=0.01,minDistance=10,blockSize=7)
        if p0 is not None and len(p0)>50:
            p1,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p0,None,**lk)
            p0r,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,p1,None,**lk)
            g=(st.ravel()==1)&(st2.ravel()==1)
            g&=np.linalg.norm((p0r-p0).reshape(-1,2),axis=1)<1.0
            if g.sum()>30:
                a=p0.reshape(-1,2)[g];b=p1.reshape(-1,2)[g]
                meds.append(np.median(np.linalg.norm(b-a,axis=1))*S); feats.append(g.sum())
        prev=cur
    if meds:
        m=np.array(meds)
        print(f'{t:>5} {np.median(m):>10.1f} {np.percentile(m,95):>7.1f} {np.median(m)*5:>14.1f} {int(np.median(feats)):>6} {int(np.median(shs)):>7}')
    else:
        print(f'{t:>5} {"FLOW FAILED":>10}')
