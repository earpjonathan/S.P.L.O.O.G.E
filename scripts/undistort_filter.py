#!/usr/bin/env python
"""Raw-frame undistortion filter: stdin bgr24 WxH -> stdout bgr24 WxH.

Applies the COLMAP OPENCV model (fx fy cx cy k1 k2 p1 p2) exactly, mapping into
a pinhole camera with the SAME intrinsics. That is the whole point: the viewer
renders a pinhole projection built from these fx/fy, so undistorting to any
other newCameraMatrix (getOptimalNewCameraMatrix and friends) would keep the
footage misaligned with the render, just differently.

Consequence worth knowing: with k1 < 0 (barrel) the output corner samples from
INSIDE the source, so the undistorted clip shows a slightly narrower field than
the raw clip -- ~12% off each side horizontally here. Nothing is stretched into
black borders; the outer sliver is simply not part of the pinhole frustum the
render draws.

ffmpeg's lenscorrection filter was the obvious alternative and is not equivalent:
it normalises radius by the half-diagonal instead of the focal length (so the
coefficients would need rescaling by (R/f)^2 and (R/f)^4) and it has no
tangential term at all.
"""
import sys
import numpy as np
import cv2

W, H = int(sys.argv[1]), int(sys.argv[2])
fx, fy, cx, cy, k1, k2, p1, p2 = map(float, sys.argv[3:11])
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
D = np.array([k1, k2, p1, p2], dtype=np.float64)
mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)

n = W * H * 3
read, write = sys.stdin.buffer.read, sys.stdout.buffer.write
while True:
    buf = read(n)               # BufferedReader.read(n) is exact-or-EOF
    if len(buf) < n:
        break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    write(cv2.remap(f, mx, my, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE).tobytes())
