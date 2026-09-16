import json, numpy as np, glob, os
def qvec2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
print('dispersion = 1 - |mean unit vector|.  0 = axis never reorients, 1 = sweeps all directions')
print(f'{"clip":<8}{"dur":>6}   {"axisX":>7}{"axisY":>7}{"axisZ":>7}   {"maxDisp":>8}   verdict')
res=[]
for f in sorted(glob.glob('gyro/ca*_*.json')):
    tag=os.path.basename(f).replace('cam_','').replace('camera_','').replace('.json','')
    d=json.load(open(f))
    q=np.array([e['org_quat'] for e in d]); ts=np.array([e['timestamp_ms'] for e in d])/1000.
    sub=slice(None,None,5)
    Rs=np.array([qvec2R(x) for x in q[sub]])
    disp=[]
    for a in range(3):
        v=Rs[:,a,:]                       # rows of R = world-frame image of camera axes
        v=v/np.linalg.norm(v,axis=1,keepdims=True)
        disp.append(1-np.linalg.norm(v.mean(0)))
    md=max(disp)
    verdict='ORBIT / wide coverage' if md>0.55 else ('partial turn' if md>0.28 else 'CORRIDOR (narrow)')
    print(f'{tag:<8}{ts[-1]-ts[0]:>5.0f}s   {disp[0]:>7.3f}{disp[1]:>7.3f}{disp[2]:>7.3f}   {md:>8.3f}   {verdict}')
    res.append((md,tag))
