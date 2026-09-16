#!/bin/bash
# Merged hilltop+cemetery reconstruction over all 7,471 images.
#
# Gated on COLMAP's own two-view verification, not a SIFT proxy: of the sampled
# cross-session pairs that verified, 6,031 carried a real baseline (CALIBRATED
# or UNCALIBRATED) against 563 PLANAR_OR_PANORAMIC -- under 1% pure rotation.
# That is the opposite of the May-vs-August case, where every cross-session
# inlier sat on the distant skyline and could not triangulate.
#
# multiple_models is left at its DEFAULT (on) deliberately. If the sessions
# join, this produces ONE model with ~7,400 images; if they do not, it produces
# two. That is the cleanest possible readout of whether the merge worked, and
# forcing a single model would only hide the answer.
set -e
P=~/Desktop/fpv-splat
OUT=$P/colmap/merged
mkdir -p "$OUT/sparse" "$OUT/snapshots"

echo "=== MERGED MAPPER $(date +%T)  ($(sqlite3 "$OUT/database.db" 'select count(*) from images') images) ==="
# snapshot_frames_freq 500, not the 200 used per-session: twice the images means
# each snapshot is far bigger, and 200 would write ~37 of them.
nice -n 5 colmap mapper \
  --database_path "$OUT/database.db" \
  --image_path $P/frames/merged \
  --output_path "$OUT/sparse" \
  --Mapper.snapshot_path "$OUT/snapshots" --Mapper.snapshot_frames_freq 500 \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 40 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.ba_global_max_refinements 2 \
  --Mapper.ba_global_frames_ratio 1.35 --Mapper.ba_global_points_ratio 1.35 \
  > $P/work/merged_map.log 2>&1
echo "=== MERGED DONE $(date +%T) ==="
ls "$OUT/sparse"
for m in "$OUT"/sparse/*/; do
  echo "--- $m ---"
  colmap model_analyzer --path "$m" 2>&1 | grep -E "Registered images|Points:|track length|reprojection"
done
