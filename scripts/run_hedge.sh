#!/bin/bash
set -e
P=~/Desktop/fpv-splat
OUT=$P/colmap/hedge26
rm -rf "$OUT"; mkdir -p "$OUT/sparse" "$OUT/snapshots"
echo "=== HEDGE MAPPER (0026 only, 1156 frames) $(date +%T) ==="
nice -n 5 colmap mapper --database_path $P/colmap/site_26.db --image_path $P/frames/site \
  --output_path "$OUT/sparse" \
  --Mapper.image_list_path $P/work/list_0026.txt \
  --Mapper.snapshot_path "$OUT/snapshots" \
  --Mapper.snapshot_frames_freq 200 \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 40 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.ba_global_max_refinements 2 \
  --Mapper.ba_global_frames_ratio 1.35 \
  --Mapper.ba_global_points_ratio 1.35 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -3
echo "=== HEDGE DONE $(date +%T) ==="
ls "$OUT/sparse"
