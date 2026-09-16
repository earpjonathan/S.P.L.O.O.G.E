#!/usr/bin/env python
"""Adaptive gyro-driven frame selection, generalised over a clip list.
kmax bounds the gap in time so translation-dominated flight (diving past
buildings, where rotation is small) still gets sampled densely enough."""
import json, numpy as np, sys, argparse
def qmul(a,b):
    w1,x1,y1,z1=a;w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def relang(a,b):
    ac=a.copy(); ac[1:]*=-1
    return 2*np.arccos(np.clip(abs(qmul(ac,b)[0]),-1,1))*180/np.pi
def select(q, thresh, kmin, kmax):
    keep=[0]; last=0
    for i in range(1,len(q)):
        gap=i-last
        if gap<kmin: continue
        if gap>=kmax or relang(q[last],q[i])>=thresh:
            keep.append(i); last=i
    return np.array(keep)
ap=argparse.ArgumentParser()
ap.add_argument('clips',nargs='+'); ap.add_argument('--thresh',type=float,default=8)
ap.add_argument('--kmin',type=int,default=2); ap.add_argument('--kmax',type=int,default=15)
ap.add_argument('--save',action='store_true')
ap.add_argument('--sweep',action='store_true')
a=ap.parse_args()
Q={c:np.array([e['org_quat'] for e in json.load(open(f'gyro/cam_{c}.json'))]) for c in a.clips}
# Guard: DJI sometimes writes the telemetry stream with org_quat identity for
# every sample (clips 0014/0015). NOT explained by codec or resolution -- the
# 2688x2016 h264 clips 0050-0060 carry perfectly good gyro, so always TEST
# rather than infer it from the recording mode. Gyroflow
# reports success and emits one sample per frame, so nothing fails loudly --
# relang() just returns 0 forever and selection silently degrades to the kmax
# time cap. Refuse rather than pretend.
for c,q in Q.items():
    if len(np.unique(q.round(6),axis=0)) < max(10, len(q)//1000):
        raise SystemExit(f'ERROR: gyro for {c} is constant '
                         f'({len(np.unique(q.round(6),axis=0))} unique quats in {len(q)}) '
                         f'-- no usable orientation, refusing to select frames from it')
if a.sweep:
    print(f'{"thresh":>7}' + ''.join(f'{c:>8}' for c in a.clips) + f'{"TOTAL":>8}')
    for th in [6,8,10,12,14,16,20]:
        ns=[len(select(Q[c],th,a.kmin,a.kmax)) for c in a.clips]
        print(f'{th:>7}' + ''.join(f'{n:>8}' for n in ns) + f'{sum(ns):>8}')
    sys.exit()
tot=0
for c in a.clips:
    k=select(Q[c],a.thresh,a.kmin,a.kmax)
    ang=np.array([relang(Q[c][k[j]],Q[c][k[j+1]]) for j in range(len(k)-1)])
    dt=np.diff(k)/50.0
    print(f'{c}: {len(k):>5} frames from {len(Q[c]):>5} ({len(k)/(len(Q[c])/50):.1f} fps)  '
          f'rot med {np.median(ang):5.2f} p99 {np.percentile(ang,99):6.2f} deg  |  '
          f'gap med {np.median(dt)*1000:.0f} ms p99 {np.percentile(dt,99)*1000:.0f} ms')
    tot+=len(k)
    if a.save: np.save(f'work/keep_{c}.npy',k)
print(f'TOTAL {tot} frames')
