import sys, numpy as np; sys.path.insert(0,'scripts')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from colmap_io import *
M='colmap/seg000/merged_ba'
ids,xyz,rgb,err,tl=read_points3D(M+'/points3D.bin')
C,R,names=centers(read_images(M+'/images.bin'))
step=np.median(np.linalg.norm(np.diff(C,axis=0),axis=1))
d=np.linalg.norm(xyz-C.mean(0),axis=1)
keep=(err<2.5)&(d/step<400)&(tl>=3)
print(f'plotting {keep.sum():,} / {len(ids):,} points (err<2.5px, <400 steps, track>=3)')
P=xyz[keep]; col=rgb[keep]/255.0
# align to camera path: PC1 = flight direction
mu=C.mean(0); A=C-mu
U,S,Vt=np.linalg.svd(A,full_matrices=False)
Pp=(P-mu)@Vt.T; Cp=A@Vt.T
fig,ax=plt.subplots(1,2,figsize=(20,7.5))
for a,(i,j),t in zip(ax,[(0,1),(0,2)],['top-down-ish (PC1 vs PC2)','side-ish (PC1 vs PC3)']):
    a.scatter(Pp[:,i],Pp[:,j],c=col,s=.7,alpha=.55,linewidths=0)
    a.plot(Cp[:,i],Cp[:,j],'-',color='red',lw=2,label='camera path')
    a.set_title(t); a.set_xlabel(f'PC{i+1}'); a.set_ylabel(f'PC{j+1}')
    a.legend(); a.grid(alpha=.25); a.set_aspect('equal','datalim')
    lo,hi=np.percentile(Pp[:,j],[1,99]); pad=(hi-lo)*.6
    a.set_ylim(lo-pad,hi+pad)
fig.suptitle('COLMAP sparse cloud + camera trajectory — seg000 merged+BA (400 frames)')
plt.tight_layout(); plt.savefig('reports/seg000_cloud.png',dpi=115); plt.close()
print('wrote reports/seg000_cloud.png')
# depth stats along viewing direction
print()
print(f'scene extent (PC units): {np.ptp(Pp[:,0]):.1f} x {np.ptp(Pp[:,1]):.1f} x {np.ptp(Pp[:,2]):.1f}')
print(f'camera path extent     : {np.ptp(Cp[:,0]):.1f} x {np.ptp(Cp[:,1]):.1f} x {np.ptp(Cp[:,2]):.1f}')
print(f'path length / lateral spread = {np.ptp(Cp[:,0])/max(np.ptp(Cp[:,1]),1e-9):.1f} : 1')
