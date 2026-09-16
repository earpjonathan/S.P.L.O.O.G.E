import numpy as np, cv2, glob, json
files=sorted(glob.glob('work/survey2/*.jpg'))
sh=[];br=[];ts=[]
for i,f in enumerate(files):
    g=cv2.imread(f,cv2.IMREAD_GRAYSCALE)
    sh.append(cv2.Laplacian(g,cv2.CV_64F).var())
    br.append(g.mean())
    ts.append(i*0.5)
sh=np.array(sh);br=np.array(br);ts=np.array(ts)
np.save('work/survey_sharp.npy',sh); np.save('work/survey_bright.npy',br)
print(f'frames {len(files)}  sharpness med {np.median(sh):.0f}  brightness med {br.mean():.0f}')
print()
# gyro yaw variation per window
d=json.load(open('gyro/camera_0049.json'))
q=np.array([e['org_quat'] for e in d]); gts=np.array([e['timestamp_ms'] for e in d])/1000.
qc=q.copy(); qc[:,1:]*=-1
def qmul(a,b):
    w1,x1,y1,z1=a.T;w2,x2,y2,z2=b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2],axis=1)
ang=2*np.arccos(np.clip(np.abs(qmul(qc[:-1],q[1:])[:,0]),-1,1))*180/np.pi

print('=== 40s WINDOWS ranked (sharpness x brightness, penalise dark) ===')
print(f'{"start":>6} {"sharp_med":>10} {"sharp_p10":>10} {"bright":>7} {"rot/s":>7} {"score":>8}')
res=[]
for s in range(0,232,10):
    m=(ts>=s)&(ts<s+40)
    if m.sum()<70: continue
    gm=(gts[:-1]>=s)&(gts[:-1]<s+40)
    smed=np.median(sh[m]); sp10=np.percentile(sh[m],10); bm=br[m].mean()
    rot=ang[gm].sum()/40
    # penalise dark (<60) and reward sharp
    score=smed*min(bm/70.0,1.0)
    res.append((score,s,smed,sp10,bm,rot))
for score,s,smed,sp10,bm,rot in sorted(res,reverse=True):
    print(f'{s:>6} {smed:>10.0f} {sp10:>10.0f} {bm:>7.1f} {rot:>7.1f} {score:>8.0f}')
