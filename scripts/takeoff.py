import numpy as np, cv2, glob
files=sorted(glob.glob('frames/seg000/*.jpg'))[:120]
lk=dict(winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
prev=cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)
prev=cv2.resize(prev,(1344,1008))
print(f'{"frame":>6} {"t(s)":>6} {"medFlow(px,full)":>17} {"sharp":>8}')
rows=[]
for i,f in enumerate(files[1:],1):
    cur=cv2.resize(cv2.imread(f,cv2.IMREAD_GRAYSCALE),(1344,1008))
    p0=cv2.goodFeaturesToTrack(prev,maxCorners=1200,qualityLevel=0.01,minDistance=12,blockSize=7)
    med=np.nan
    if p0 is not None and len(p0)>50:
        p1,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p0,None,**lk)
        p0r,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,p1,None,**lk)
        g=(st.ravel()==1)&(st2.ravel()==1)
        g&=np.linalg.norm((p0r-p0).reshape(-1,2),axis=1)<1.0
        if g.sum()>30:
            a=p0.reshape(-1,2)[g];b=p1.reshape(-1,2)[g]
            med=np.median(np.linalg.norm(b-a,axis=1))*2
    sh=cv2.Laplacian(cur,cv2.CV_64F).var()
    rows.append((i,i/10.0,med,sh))
    if i%5==0 or i<20:
        print(f'{i:>6} {i/10.0:>6.1f} {med:>17.2f} {sh:>8.0f}')
r=np.array(rows)
mv=r[:,2]
moving=np.where(mv>3.0)[0]
print()
print(f'first frame with flow >3 px: index {rows[moving[0]][0]} at t={rows[moving[0]][1]:.1f}s' if len(moving) else 'never moves')
still=(mv<1.5).sum()
print(f'frames with flow <1.5 px (essentially static): {still} of {len(mv)}')
