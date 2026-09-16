#!/usr/bin/env python
import json, sys
import numpy as np
sys.path.insert(0,'/Users/jonathanearp/Desktop/fpv-splat/scripts')
from raster import load_splat, sigma_world, render
from PIL import Image

SPLAT, CAMS, IDX, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
margin = float(sys.argv[5]) if len(sys.argv) > 5 else 1.2
W, H = 640, 480
xyz, sc, rgba, rot = load_splat(SPLAT)
S3 = sigma_world(sc, rot)
cam = json.load(open(CAMS))[IDX]
img, acc = render(xyz, S3, rgba, cam, W, H, margin=margin)
# left: colour on mid-grey.  right: accumulated opacity (red = see-through)
panel = np.zeros((H, W*2, 3))
panel[:, :W] = img + (1-acc)[:,:,None]*np.array([0.5,0.5,0.55])
o = acc[:,:,None]
panel[:, W:] = o*np.array([1,1,1]) + (1-o)*np.array([1,0.1,0.1])
for x in (W,):
    panel[:, x-1:x+1] = 0
for y in (H//3, 2*H//3):
    panel[y-1:y+1, :] = [0,0.6,1]
    panel[y-1:y+1, W:] = [0,0.6,1]
Image.fromarray((panel*255).astype(np.uint8)).save(OUT)
b = acc[2*H//3:]
print(f"{cam['img_name']} margin={margin}  bottom-3rd opacity {b.mean():.3f} "
      f" holes<0.5 {100*np.mean(b<0.5):.1f}%  -> {OUT}")
