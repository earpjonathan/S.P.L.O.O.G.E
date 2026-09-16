#!/bin/bash
set -e
P=~/Desktop/fpv-splat; OUT=$P/colmap/site; IMG=$P/frames/site
rm -rf "$OUT"; mkdir -p "$OUT/sparse"
echo "=== FEATURES $(date +%T) ==="
colmap feature_extractor --database_path "$OUT/database.db" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "733,733,960,720,-0.1066,0.0077,0.0013,0.0004" \
  --FeatureExtraction.use_gpu 0 2>&1 | grep -E "Elapsed|ERROR" | tail -2
echo "=== SEQUENTIAL MATCH $(date +%T) ==="
colmap sequential_matcher --database_path "$OUT/database.db" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 12 --SequentialMatching.quadratic_overlap 1 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -2
echo "=== CROSS-CLIP MATCH $(date +%T) ==="
cd $P && $P/.venv/bin/python scripts/gen_cross_pairs.py 8
colmap matches_importer --database_path "$OUT/database.db" \
  --match_list_path "$P/work/cross_pairs.txt" --match_type pairs \
  --FeatureMatching.use_gpu 0 2>&1 | grep -E "Elapsed|ERROR" | tail -2
echo "=== MAPPER $(date +%T) ==="
colmap mapper --database_path "$OUT/database.db" --image_path "$IMG" \
  --output_path "$OUT/sparse" \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 40 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -3
echo "=== SITE DONE $(date +%T) ==="
ls "$OUT/sparse"
