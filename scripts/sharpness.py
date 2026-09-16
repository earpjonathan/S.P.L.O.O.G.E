#!/usr/bin/env python
"""Bottom-third DETAIL loss: render vs ground truth, from a Brush eval dir.

The hole metric measures whether a ray accumulates enough alpha. It says
nothing about whether the ground carries TEXTURE. A render can be fully opaque
and still read as a smeared mass, which is a different defect with a different
cause. This measures high-frequency energy (mean |grad|) in the bottom third,
render vs GT, so the two failure modes can be told apart.
"""
import sys, os, glob
import numpy as np
from PIL import Image

evd = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rows = []
for f in sorted(glob.glob(os.path.join(evd, '*.png')))[:lim]:
    name = os.path.basename(f)[:-4]
    # Prefer the GT set whose size MATCHES the render. Taking the first
    # alphabetical match silently paired a 2688 render with the 1920 GT and
    # then upscaled the GT to fit -- scoring a real 2688 render against a
    # reference carrying no detail above 1920's Nyquist, which flatters it.
    cands = [os.path.join(fs, name) for fs in sorted(glob.glob('frames/*'))
             if os.path.exists(os.path.join(fs, name))]
    gt = None
    if cands:
        rsize = Image.open(f).size
        exact = [c for c in cands if Image.open(c).size == rsize]
        gt = exact[0] if exact else cands[0]
    if gt is None: continue
    r = Image.open(f).convert('L'); g = Image.open(gt).convert('L')
    if g.size != r.size: g = g.resize(r.size, Image.LANCZOS)
    a = np.asarray(r, np.float64)/255.0; b = np.asarray(g, np.float64)/255.0
    H = a.shape[0]; lo = 2*H//3
    def hf(x):
        gx = np.abs(np.diff(x, axis=1)).mean(); gy = np.abs(np.diff(x, axis=0)).mean()
        return (gx+gy)/2
    rows.append((hf(a[lo:]), hf(b[lo:]), hf(a[:H//2]), hf(b[:H//2])))
rows = np.array(rows)
if len(rows)==0: print('no GT matched'); sys.exit()
rb, gb, rt, gt_ = rows.mean(0)
print(f'  n={len(rows)}')
print(f'  BOTTOM third  render {rb:.4f}  gt {gb:.4f}   detail retained {100*rb/gb:5.1f}%')
print(f'  TOP half      render {rt:.4f}  gt {gt_:.4f}   detail retained {100*rt/gt_:5.1f}%')
