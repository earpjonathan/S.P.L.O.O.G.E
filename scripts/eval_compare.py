#!/usr/bin/env python
"""Side-by-side render vs ground-truth montage + PSNR for a Brush eval dir."""
import sys, os, glob, numpy as np
from PIL import Image, ImageDraw

evd = sys.argv[1]; out = sys.argv[2]
picks = sys.argv[3].split(',') if len(sys.argv) > 3 else None
files = sorted(glob.glob(os.path.join(evd, '*.png')))
if picks:
    files = [f for f in files if any(p in os.path.basename(f) for p in picks)]

rows, scores = [], []
W = 640
for f in files:
    name = os.path.basename(f)[:-4]                 # strip .png -> 0026_XXXX.jpg
    # Search every frameset rather than the two this script was written for:
    # a hardcoded list silently reports "images: 0" on any newer scene, which
    # reads as a result rather than as a missing-file error.
    gt_p = None
    for _fs in sorted(glob.glob('frames/*')):
        _c = os.path.join(_fs, name)
        if os.path.exists(_c): gt_p = _c; break
    if gt_p is None: continue
    r = Image.open(f).convert('RGB'); g = Image.open(gt_p).convert('RGB')
    if g.size != r.size: g = g.resize(r.size, Image.LANCZOS)
    a = np.asarray(r, np.float64)/255.0; b = np.asarray(g, np.float64)/255.0
    mse = float(((a-b)**2).mean())
    psnr = 10*np.log10(1.0/mse) if mse > 0 else 99.0
    scores.append((name, psnr))
    h = int(r.height * W / r.width)
    rows.append((r.resize((W,h), Image.LANCZOS), g.resize((W,h), Image.LANCZOS), name, psnr))

if rows:
    w, h = rows[0][0].size
    canvas = Image.new('RGB', (w*2, h*len(rows)+22*len(rows)), (16,16,18))
    d = ImageDraw.Draw(canvas)
    for i,(r,g,name,psnr) in enumerate(rows):
        y = i*(h+22)
        d.text((6, y+5), f'{name}   RENDER (iter eval)   PSNR {psnr:.2f} dB', fill=(235,235,240))
        d.text((w+6, y+5), 'GROUND TRUTH', fill=(235,235,240))
        canvas.paste(r, (0, y+22)); canvas.paste(g, (w, y+22))
    canvas.save(out, quality=92)

ps = [s for _,s in scores]
print(f'images   : {len(ps)}')
if ps:
    print(f'PSNR mean: {np.mean(ps):.2f} dB   median {np.median(ps):.2f}   min {np.min(ps):.2f}   max {np.max(ps):.2f}')
    for n,s in sorted(scores, key=lambda x:x[1])[:3]: print(f'  worst {n}: {s:.2f} dB')
    for n,s in sorted(scores, key=lambda x:-x[1])[:3]: print(f'  best  {n}: {s:.2f} dB')
