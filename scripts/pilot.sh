#!/bin/bash
set -e
NAME=$1; MODEL=$2; PARAMS=$3
P=~/Desktop/fpv-splat; OUT=$P/colmap/$NAME
rm -rf "$OUT"; mkdir -p "$OUT/sparse"
echo "### $NAME  model=$MODEL params=$PARAMS"
colmap feature_extractor --database_path "$OUT/database.db" --image_path "$P/frames/pilot26" \
  --ImageReader.camera_model "$MODEL" --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$PARAMS" --FeatureExtraction.use_gpu 0 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -2
colmap sequential_matcher --database_path "$OUT/database.db" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 12 --SequentialMatching.quadratic_overlap 1 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -2
colmap mapper --database_path "$OUT/database.db" --image_path "$P/frames/pilot26" \
  --output_path "$OUT/sparse" \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 25 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  2>&1 | grep -E "Elapsed|ERROR" | tail -2
echo "### $NAME DONE"
