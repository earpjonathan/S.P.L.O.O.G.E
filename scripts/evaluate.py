import numpy as np, json, os, sys, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'scripts')
from colmap_io import *

SEG = sys.argv[1] if len(sys.argv)>1 else 'seg000'
TAG = ''
FPS_EXTRACT = float(sys.argv[2]) if len(sys.argv)>2 else 10.0
SEG_START   = float(sys.argv[3]) if len(sys.argv)>3 else 0.0
MODEL_OVERRIDE = sys.argv[4] if len(sys.argv)>4 else None
base=f'colmap/{SEG}/sparse'
models=[MODEL_OVERRIDE] if MODEL_OVERRIDE else sorted(glob.glob(base+'/*'))
if MODEL_OVERRIDE: TAG='_'+MODEL_OVERRIDE.rstrip('/').split('/')[-1]
if not models: print('NO MODEL PRODUCED'); sys.exit(1)
print(f'models produced: {len(models)}  -> {[os.path.basename(m) for m in models]}')
# pick largest
best=None;bestn=-1
for m in models:
    try:
        im=read_images(m+'/images.bin')
        if len(im)>bestn: bestn=len(im); best=m
    except Exception as e: print(' skip',m,e)
print(f'largest model: {best} with {bestn} registered images')
imgs=read_images(best+'/images.bin')
cams=read_cameras(best+'/cameras.bin')
ids,xyz,rgb,err,tl=read_points3D(best+'/points3D.bin')
N_INPUT=len(glob.glob(f'frames/{SEG}/*.jpg'))

print()
print('='*64)
print(f'  SEGMENT {SEG}   input frames: {N_INPUT}')
print('='*64)
reg=len(imgs)
print(f'  registered images      : {reg} / {N_INPUT}  = {100*reg/N_INPUT:.1f}%   [target >85%]')
print(f'  sparse 3D points       : {len(ids):,}')
print(f'  mean reprojection error: {err.mean():.4f} px   (median {np.median(err):.4f})')
print(f'  mean track length      : {tl.mean():.2f}   (median {np.median(tl):.0f})')
obs=sum(int((imgs[k]['p3d']>=0).sum()) for k in imgs)
print(f'  mean observations/image: {obs/reg:.1f}')
for cid,c in cams.items():
    MODELS={0:'SIMPLE_PINHOLE',1:'PINHOLE',2:'SIMPLE_RADIAL',3:'RADIAL',4:'OPENCV',5:'OPENCV_FISHEYE'}
    print(f'  camera #{cid} {MODELS.get(c["model"],c["model"])} {c["w"]}x{c["h"]}')
    p=c['params']
    print(f'     fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.2f} cy={p[3]:.2f}')
    if len(p)>=8: print(f'     k1={p[4]:.5f} k2={p[5]:.5f} p1={p[6]:.6f} p2={p[7]:.6f}')
    hf=2*np.degrees(np.arctan(c['w']/2/p[0])); df=2*np.degrees(np.arctan(np.hypot(c['w'],c['h'])/2/p[0]))
    print(f'     => HFOV {hf:.1f} deg, DFOV {df:.1f} deg   (prior was 965.40 / 108.6 / 120.2)')

C,R,names=centers(imgs)
# which input frames registered
idx=np.array([int(os.path.splitext(n)[0])-1 for n in names])
gaps=np.diff(np.sort(idx))
print(f'  largest gap in registered sequence: {gaps.max() if len(gaps) else 0} frames')
print(f'  number of breaks (gap>1)          : {int((gaps>1).sum())}')

# ---------- trajectory ----------
Cc=C-C.mean(0)
U,S,Vt=np.linalg.svd(Cc,full_matrices=False)
P=Cc@Vt.T
print()
print(f'  trajectory PCA extent (model units): {np.ptp(P[:,0]):.2f} x {np.ptp(P[:,1]):.2f} x {np.ptp(P[:,2]):.2f}')
print(f'  PCA singular values: {S/S[0]}')
step=np.linalg.norm(np.diff(P,axis=0),axis=1)
print(f'  inter-frame step: median {np.median(step):.4f}  p95 {np.percentile(step,95):.4f}  max {step.max():.4f}')
print(f'  step outliers (>5x median): {int((step>5*np.median(step)).sum())}')

order=np.argsort(idx)
fig,ax=plt.subplots(1,3,figsize=(19,5.6))
lbl=['PC1','PC2','PC3']
for a,(i,j) in zip(ax,[(0,1),(0,2),(1,2)]):
    sc=a.scatter(P[order,i],P[order,j],c=idx[order]/FPS_EXTRACT+SEG_START,cmap='viridis',s=13,zorder=3)
    a.plot(P[order,i],P[order,j],'-',lw=0.7,color='0.5',zorder=2)
    a.set_xlabel(lbl[i]);a.set_ylabel(lbl[j]);a.set_aspect('equal','datalim');a.grid(alpha=.3)
    a.set_title(f'{lbl[i]} vs {lbl[j]}')
plt.colorbar(sc,ax=ax[2],label='time (s)')
fig.suptitle(f'COLMAP camera trajectory — {SEG} — {reg}/{N_INPUT} frames registered ({100*reg/N_INPUT:.1f}%)')
plt.tight_layout();plt.savefig(f'reports/{SEG}{TAG}_trajectory.png',dpi=125);plt.close()
print(f'  wrote reports/{SEG}{TAG}_trajectory.png')

# ---------- gyro vs colmap rotation ----------
d=json.load(open('gyro/camera_0049.json'))
gq=np.array([e['org_quat'] for e in d]); gts=np.array([e['timestamp_ms'] for e in d])/1000.
# map registered image -> source video frame
src=np.round((SEG_START+idx/FPS_EXTRACT)*50).astype(int)
src=np.clip(src,0,len(gq)-1)
Rg=np.array([qvec2R(gq[s]) for s in src])
Rc=R  # world->cam
o=np.argsort(idx); Rc=Rc[o];Rg=Rg[o];idxs=idx[o];srcs=src[o]
def ang_of(M):
    return np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)))
consec=np.where(np.diff(idxs)==1)[0]
ac=[];ag=[]
for i in consec:
    ac.append(ang_of(Rc[i+1]@Rc[i].T))
    ag.append(ang_of(Rg[i+1]@Rg[i].T))
ac=np.array(ac);ag=np.array(ag)
resid=ac-ag
print()
print(f'  --- rotation cross-check (consecutive pairs, n={len(ac)}) ---')
print(f'  COLMAP inter-frame rotation: median {np.median(ac):.3f} deg')
print(f'  GYRO   inter-frame rotation: median {np.median(ag):.3f} deg')
print(f'  difference: median {np.median(resid):+.4f}  MAD {1.4826*np.median(np.abs(resid-np.median(resid))):.4f}  RMS {np.sqrt((resid**2).mean()):.4f} deg')
cc=np.corrcoef(ac,ag)[0,1]
print(f'  correlation COLMAP vs gyro rotation magnitude: r = {cc:.4f}')

t=(SEG_START+idxs/FPS_EXTRACT)
fig,ax=plt.subplots(2,1,figsize=(15,8),height_ratios=[2,1])
ax[0].plot(t[consec],ag,label='Gyroflow (DJI IMU)',lw=1.5)
ax[0].plot(t[consec],ac,label='COLMAP',lw=1.1,alpha=.85)
ax[0].set_ylabel('inter-frame rotation (deg)');ax[0].legend();ax[0].grid(alpha=.3)
ax[0].set_title(f'Camera rotation per 0.1s step — COLMAP vs gyro   (r={cc:.4f})')
ax[1].plot(t[consec],resid,lw=.9,color='crimson')
ax[1].axhline(0,color='k',lw=.6);ax[1].set_ylabel('COLMAP - gyro (deg)')
ax[1].set_xlabel('time (s)');ax[1].grid(alpha=.3)
plt.tight_layout();plt.savefig(f'reports/{SEG}{TAG}_rotation_check.png',dpi=125);plt.close()
print(f'  wrote reports/{SEG}{TAG}_rotation_check.png')

json.dump(dict(seg=SEG,n_input=N_INPUT,n_reg=reg,pct=100*reg/N_INPUT,
               n_points=int(len(ids)),mean_reproj=float(err.mean()),
               mean_track=float(tl.mean()),rot_corr=float(cc),
               rot_rms=float(np.sqrt((resid**2).mean())),n_models=len(models)),
          open(f'reports/{SEG}{TAG}_metrics.json','w'),indent=2)
print('  wrote metrics json')
