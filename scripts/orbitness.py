import json, numpy as np, glob, os
from scipy.spatial.transform import Rotation as R
print(f'{"clip":<12}{"dur":>6}{"totYaw":>9}{"netYaw":>9}{"hdg cover":>11}{"rot/s":>8}   {"read"}')
for f in sorted(glob.glob('gyro/ca*_*.json')):
    tag=os.path.basename(f).replace('cam_','').replace('camera_','').replace('.json','')
    d=json.load(open(f))
    q=np.array([e['org_quat'] for e in d]); ts=np.array([e['timestamp_ms'] for e in d])/1000.
    rot=R.from_quat(np.column_stack([q[:,1],q[:,2],q[:,3],q[:,0]]))
    eul=rot.as_euler('ZYX',degrees=True)
    yaw=eul[:,0]
    yawu=np.unwrap(np.radians(yaw))*180/np.pi
    tot=np.abs(np.diff(yawu)).sum(); net=yawu[-1]-yawu[0]
    # heading coverage: fraction of 36 x 10-deg compass bins visited
    bins=np.zeros(36,bool); bins[((yaw+180)//10).astype(int)%36]=True
    cover=bins.mean()
    qc=q.copy(); qc[:,1:]*=-1
    def qmul(a,b):
        w1,x1,y1,z1=a.T;w2,x2,y2,z2=b.T
        return np.stack([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                         w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2],axis=1)
    ang=2*np.arccos(np.clip(np.abs(qmul(qc[:-1],q[1:])[:,0]),-1,1))*180/np.pi
    dur=ts[-1]-ts[0]
    verdict = 'ORBIT-LIKE' if cover>0.75 else ('mixed' if cover>0.45 else 'CORRIDOR')
    print(f'{tag:<12}{dur:>5.0f}s{tot:>9.0f}{net:>9.0f}{100*cover:>10.0f}%{ang.sum()/dur:>8.1f}   {verdict}')
