#!/bin/bash
set -e
PROJ=~/Desktop/fpv-splat
SEG=${1:-seg000}
OUT=$PROJ/colmap/$SEG
IMG=$PROJ/frames/$SEG
rm -rf "$OUT/sparse_tuned"; mkdir -p "$OUT/sparse_tuned"
echo "=========== TUNED MAPPER $(date) ==========="
# Forward-motion FPV video: default init gates reject almost every candidate pair.
#   init_min_tri_angle 16 -> 3     (forward motion gives tiny triangulation angles)
#   init_max_forward_motion 0.95 -> 1.0  (stop rejecting forward-dominated pairs)
#   min_model_size 10 -> 25        (don't emit junk fragments)
colmap mapper --database_path "$OUT/database.db" --image_path "$IMG" \
  --output_path "$OUT/sparse_tuned" \
  --Mapper.init_min_tri_angle 3 \
  --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 \
  --Mapper.init_min_num_inliers 80 \
  --Mapper.min_model_size 25 \
  --Mapper.filter_min_tri_angle 1.0 \
  --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.abs_pose_min_inlier_ratio 0.2 \
  2>&1 | grep -E "Registering|Registered|=> |Elapsed|ERROR" | tail -80
echo "=========== TUNED DONE $(date) ==========="
ls "$OUT/sparse_tuned"
