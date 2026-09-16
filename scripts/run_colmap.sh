#!/bin/bash
set -e
PROJ=~/Desktop/fpv-splat
SEG=${1:-seg000}
OUT=$PROJ/colmap/$SEG
IMG=$PROJ/frames/$SEG
rm -rf "$OUT"; mkdir -p "$OUT/sparse"
echo "=========== FEATURE EXTRACTION $(date) ==========="
colmap feature_extractor \
  --database_path "$OUT/database.db" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "965.4,965.4,1344.0,1008.0,0,0,0,0" \
  --FeatureExtraction.use_gpu 0 2>&1 | grep -E "Processed|Elapsed|ERROR" | tail -5
echo "=========== MATCHING $(date) ==========="
colmap sequential_matcher --database_path "$OUT/database.db" \
  --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 10 --SequentialMatching.quadratic_overlap 1 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -3
echo "=========== MAPPER $(date) ==========="
colmap mapper --database_path "$OUT/database.db" --image_path "$IMG" \
  --output_path "$OUT/sparse" 2>&1 | grep -E "Registering|Registered|=> |Elapsed|ERROR|Bundle" | tail -60
echo "=========== DONE $(date) ==========="
ls -la "$OUT/sparse"/*/ 2>&1
