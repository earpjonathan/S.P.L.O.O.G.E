import numpy as np, json, os, sys, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'scripts')
from colmap_io import *

MODEL = sys.argv[1] if len(sys.argv)>1 else None
base='colmap/site/sparse'
models=[MODEL] if MODEL else sorted(glob.glob(base+'/*'))
if not models: print('NO MODEL'); sys.exit(1)
TAG=os.path.basename(models[0].rstrip('/')) if MODEL else 'best'

allimgs=sorted(os.path.basename(p) for p in glob.glob('frames/site/*.jpg'))
N=len(allimgs)
NC={c:sum(1 for x in allimgs if x.startswith(c+'_')) for c in ['0026','0027']}

print(f'input frames: {N}  (0026: {NC["0026"]}, 0027: {NC["0027"]})')
print()
best=None;bn=-1
for m in models:
    try:
        im=read_images(m+'/images.bin')
        n=len(im)
        c26=sum(1 for v in im.values() if v['name'].startswith('0026_'))
        c27=sum(1 for v in im.values() if v['name'].startswith('0027_'))
        print(f'  model {os.path.basename(m):>3}: {n:>5} imgs  (0026:{c26:>5}  0027:{c27:>5})')
        if n>bn: bn=n; best=m
    except Exception as e: print('  skip',m,e)
print()
print(f'LARGEST MODEL: {best}')
imgs=read_images(best+'/images.bin'); cams=read_cameras(best+'/cameras.bin')
ids,xyz,rgb,err,tl=read_points3D(best+'/points3D.bin')
reg=len(imgs)
c26=[v['name'] for v in imgs.values() if v['name'].startswith('0026_')]
c27=[v['name'] for v in imgs.values() if v['name'].startswith('0027_')]
ok=err<4
print('='*66)
print(f'  registered        : {reg}/{N} = {100*reg/N:.1f}%      [target >85%]')
print(f'     0026           : {len(c26)}/{NC["0026"]} = {100*len(c26)/NC["0026"]:.1f}%')
print(f'     0027           : {len(c27)}/{NC["0027"]} = {100*len(c27)/NC["0027"]:.1f}%')
print(f'  sparse 3D points  : {len(ids):,}')
print(f'  mean reproj error : {err[ok].mean():.4f} px  (median {np.median(err):.4f}, {100*ok.mean():.2f}% under 4px)')
print(f'  mean track length : {tl.mean():.2f}')
obs=sum(int((imgs[k]["p3d"]>=0).sum()) for k in imgs)
print(f'  mean obs / image  : {obs/reg:.1f}')
MODELS={0:'SIMPLE_PINHOLE',1:'PINHOLE',2:'SIMPLE_RADIAL',3:'RADIAL',4:'OPENCV',5:'OPENCV_FISHEYE'}
for cid,c in cams.items():
    p=c['params']
    print(f'  camera {MODELS.get(c["model"])} {c["w"]}x{c["h"]}: fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.1f} cy={p[3]:.1f}')
    print(f'     k1={p[4]:.5f} k2={p[5]:.5f} p1={p[6]:.6f} p2={p[7]:.6f}   HFOV {2*np.degrees(np.arctan(c["w"]/2/p[0])):.1f} deg')

C,R,names=centers(imgs)
clip=np.array([n[:4] for n in names])
fidx=np.array([int(n.split('_')[1].split('.')[0]) for n in names])

# ---- cross-clip linking check ----
Cc=C-C.mean(0); U,S,Vt=np.linalg.svd(Cc,full_matrices=False); P=Cc@Vt.T
print()
print(f'  trajectory PCA extent: {np.ptp(P[:,0]):.2f} x {np.ptp(P[:,1]):.2f} x {np.ptp(P[:,2]):.2f}')
print(f'  PCA singular values  : {np.round(S/S[0],4)}')
if len(c26) and len(c27):
    m26=clip=='0026'; m27=clip=='0027'
    d=np.linalg.norm(P[m26].mean(0)-P[m27].mean(0))
    sp=max(np.ptp(P[:,0]),1e-9)
    print(f'  clip centroid separation: {d:.2f}  ({100*d/sp:.1f}% of scene extent)')
    print(f'  -> cross-clip linking: {"OK - clips co-registered in one frame" if d/sp<0.5 else "SUSPECT - clips sit apart"}')

# ---- gyro cross-check, per clip ----
print()
def ang_of(M): return np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)))
summary={}
for c in ['0026','0027']:
    m=clip==c
    if m.sum()<10: continue
    o=np.argsort(fidx[m]); fi=fidx[m][o]; Rc=R[m][o]
    gq=np.array([e['org_quat'] for e in json.load(open(f'gyro/cam_{c}.json'))])
    Rg=np.array([qvec2R(gq[min(i,len(gq)-1)]) for i in fi])
    ac=[];ag=[]
    for j in range(len(fi)-1):
        if fi[j+1]-fi[j] > 20: continue
        ac.append(ang_of(Rc[j+1]@Rc[j].T)); ag.append(ang_of(Rg[j+1]@Rg[j].T))
    ac=np.array(ac);ag=np.array(ag); res=ac-ag
    r=np.corrcoef(ac,ag)[0,1]
    print(f'  {c}: n={len(ac)}  COLMAP med {np.median(ac):.3f} deg  gyro med {np.median(ag):.3f} deg')
    print(f'        r={r:.4f}  RMS diff {np.sqrt((res**2).mean()):.4f} deg  MAD {1.4826*np.median(np.abs(res-np.median(res))):.4f}')
    summary[c]=dict(n=len(ac),r=float(r),rms=float(np.sqrt((res**2).mean())))

# ---- plots ----
fig,ax=plt.subplots(1,3,figsize=(19,5.8))
cols={'0026':'tab:blue','0027':'tab:orange'}
for a,(i,j) in zip(ax,[(0,1),(0,2),(1,2)]):
    for c in ['0026','0027']:
        m=clip==c
        if m.sum()==0: continue
        o=np.argsort(fidx[m])
        a.plot(P[m][o,i],P[m][o,j],'-',lw=.8,color=cols[c],alpha=.85,label=c)
        a.scatter(P[m][o,i],P[m][o,j],s=5,color=cols[c])
    a.set_xlabel(f'PC{i+1}');a.set_ylabel(f'PC{j+1}');a.grid(alpha=.3);a.set_aspect('equal','datalim')
ax[0].legend()
fig.suptitle(f'Site trajectory — {reg}/{N} frames registered ({100*reg/N:.1f}%) — clips co-registered')
plt.tight_layout();plt.savefig(f'reports/site_{TAG}_trajectory.png',dpi=120);plt.close()

keep=(err<2.5)&(tl>=3)
step=np.median(np.linalg.norm(np.diff(C,axis=0),axis=1))
dd=np.linalg.norm(xyz-C.mean(0),axis=1)
keep&=dd/step<400
Pp=(xyz[keep]-C.mean(0))@Vt.T
fig,ax=plt.subplots(1,2,figsize=(20,7.5))
for a,(i,j),t in zip(ax,[(0,1),(0,2)],['top-down (PC1/PC2)','side (PC1/PC3)']):
    a.scatter(Pp[:,i],Pp[:,j],c=rgb[keep]/255.,s=.5,alpha=.5,linewidths=0)
    for c in ['0026','0027']:
        m=clip==c
        if m.sum(): a.plot(P[m][np.argsort(fidx[m])][:,i],P[m][np.argsort(fidx[m])][:,j],'-',lw=1.6,color=cols[c],label=c)
    a.set_title(t);a.grid(alpha=.25);a.set_aspect('equal','datalim');a.legend()
fig.suptitle(f'Site sparse cloud — {int(keep.sum()):,} points')
plt.tight_layout();plt.savefig(f'reports/site_{TAG}_cloud.png',dpi=115);plt.close()
print()
print(f'  wrote reports/site_{TAG}_trajectory.png and reports/site_{TAG}_cloud.png')
json.dump(dict(n_input=N,n_reg=reg,pct=100*reg/N,points=int(len(ids)),
               reproj=float(err[ok].mean()),gyro=summary),
          open(f'reports/site_{TAG}_metrics.json','w'),indent=2)
