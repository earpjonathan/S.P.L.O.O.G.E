#!/bin/bash
# Build the viewer's footage-overlay proxy for one clip, undistorted to match
# the render.
#   make_proxy.sh <clip> <src.MP4> <fx> <fy> <cx> <cy> <k1> <k2> <p1> <p2>
# Intrinsics are the COLMAP OPENCV params AT THE SOURCE FRAME SIZE used for
# reconstruction (1920x1440 here); this scales them to the 640x480 proxy and
# shifts the principal point by the half-pixel COLMAP/OpenCV convention gap
# (COLMAP puts the image centre at W/2, OpenCV at (W-1)/2).
#
# Writes to work/ and mv's into place only on success: -movflags +faststart
# puts the moov atom LAST, so a half-written file is not merely truncated, it
# is unplayable, and the viewer reports NETWORK_NO_SOURCE.
set -e
P=~/Desktop/fpv-splat
CLIP=$1; SRC=$2; shift 2
W=640; H=480
FX=$(echo "$1/3" | bc -l); FY=$(echo "$2/3" | bc -l)
CX=$(echo "$3/3-0.5" | bc -l); CY=$(echo "$4/3-0.5" | bc -l)
K1=$5; K2=$6; P1=$7; P2=$8
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$SRC")

echo "[$CLIP] $SRC -> ${W}x${H}  K=($FX $FY $CX $CY)  D=($K1 $K2 $P1 $P2)  fps=$FPS"
nice -n 10 ffmpeg -hide_banner -loglevel error -hwaccel videotoolbox -i "$SRC" \
    -vf scale=$W:$H:flags=lanczos -f rawvideo -pix_fmt bgr24 - \
  | nice -n 10 $P/.venv/bin/python $P/scripts/undistort_filter.py $W $H \
        $FX $FY $CX $CY $K1 $K2 $P1 $P2 \
  | nice -n 10 ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt bgr24 \
        -s ${W}x${H} -r "$FPS" -i - -c:v libx264 -preset medium -crf 30 \
        -pix_fmt yuv420p -an -movflags +faststart "$P/work/${CLIP}_proxy.mp4"
mv "$P/work/${CLIP}_proxy.mp4" "$P/viewer/video/${CLIP}.mp4"
echo "[$CLIP] done $(date +%T)  $(ls -lh $P/viewer/video/${CLIP}.mp4 | awk '{print $5}')"
