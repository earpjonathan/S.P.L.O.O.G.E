import json, numpy as np, sys
def qmul(a,b):
    w1,x1,y1,z1=a;w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def relang(a,b):
    ac=a.copy(); ac[1:]*=-1
    return 2*np.arccos(np.clip(abs(qmul(ac,b)[0]),-1,1))*180/np.pi

def select(clip, thresh, kmin, kmax):
    d=json.load(open(f'gyro/cam_{clip}.json'))
    q=np.array([e['org_quat'] for e in d])
    keep=[0]; last=0
    for i in range(1,len(q)):
        gap=i-last
        if gap<kmin: continue
        if gap>=kmax or relang(q[last],q[i])>=thresh:
            keep.append(i); last=i
    keep=np.array(keep)
    ang=np.array([relang(q[keep[j]],q[keep[j+1]]) for j in range(len(keep)-1)])
    return keep,ang

TARGET=int(sys.argv[1]) if len(sys.argv)>1 else 1400
best=None
for th in [4,5,6,7,8,9,10,12,14,16]:
    tot=0; res={}
    for c in ['0026','0027']:
        k,a=select(c,th,2,15); res[c]=(k,a); tot+=len(k)
    if best is None or abs(tot-TARGET)<abs(best[1]-TARGET): best=(th,tot,res)
    print(f'thresh {th:>3} deg -> {tot:>5} frames total')
th,tot,res=best
print()
print(f'CHOSEN threshold {th} deg -> {tot} frames')
for c in ['0026','0027']:
    k,a=res[c]
    fps=len(k)/(len(json.load(open(f"gyro/cam_{c}.json")))/50)
    print(f'  {c}: {len(k):>4} frames (avg {fps:.1f} fps)  inter-frame rot: median {np.median(a):.2f} p90 {np.percentile(a,90):.2f} p99 {np.percentile(a,99):.2f} max {a.max():.2f} deg')
    print(f'        px shift @1920 (f~700/rad): median {np.median(a)*700*np.pi/180:.0f}  p99 {np.percentile(a,99)*700*np.pi/180:.0f}')
    np.save(f'work/keep_{c}.npy', k)
print()
print('saved work/keep_0026.npy, work/keep_0027.npy')
