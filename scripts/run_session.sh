#!/bin/bash
# Generic single-session COLMAP reconstruction.
#   run_session.sh <frameset> <colmap_subdir> "<clip list>" [cross_stride]
# e.g. run_session.sh aug2 aug2 "0057 0058 0059 0060"
#
# scripts/run_aug.sh is the hilltop instance of this, kept separate because it
# was already queued when this was written. New sessions should use this.
set -e
P=~/Desktop/fpv-splat
SET=$1; SUB=$2; CLIPS=$3; STRIDE=${4:-10}
OUT=$P/colmap/$SUB; IMG=$P/frames/$SET
rm -rf "$OUT"; mkdir -p "$OUT/sparse" "$OUT/snapshots"

echo "=== [$SUB] FEATURES $(date +%T)  ($(ls $IMG | wc -l | tr -d ' ') images) ==="
# Seeded from the site's converged OPENCV params. Valid for any FOV:Normal
# clip on this O4P regardless of recording resolution -- the FOV tag decides
# the field of view, not the pixel count.
colmap feature_extractor --database_path "$OUT/database.db" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "725,724,960,720,-0.107,0.008,0.001,0" \
  --FeatureExtraction.use_gpu 0 > $P/work/${SUB}_feat.log 2>&1
grep -E "Elapsed" $P/work/${SUB}_feat.log | tail -1

echo "=== [$SUB] SEQUENTIAL MATCH $(date +%T) ==="
colmap sequential_matcher --database_path "$OUT/database.db" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 12 --SequentialMatching.quadratic_overlap 1 \
  > $P/work/${SUB}_seq.log 2>&1
grep -E "Elapsed" $P/work/${SUB}_seq.log | tail -1

echo "=== [$SUB] CROSS-CLIP MATCH $(date +%T) ==="
cd $P && $P/.venv/bin/python scripts/gen_cross_pairs2.py $SET $CLIPS \
  --stride $STRIDE --out $P/work/cross_${SUB}.txt
colmap matches_importer --database_path "$OUT/database.db" \
  --match_list_path "$P/work/cross_${SUB}.txt" --match_type pairs \
  --FeatureMatching.use_gpu 0 > $P/work/${SUB}_cross.log 2>&1
grep -E "Elapsed" $P/work/${SUB}_cross.log | tail -1

echo "=== [$SUB] MAPPER $(date +%T) ==="
nice -n 5 colmap mapper --database_path "$OUT/database.db" --image_path "$IMG" \
  --output_path "$OUT/sparse" \
  --Mapper.snapshot_path "$OUT/snapshots" --Mapper.snapshot_frames_freq 200 \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 40 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.ba_global_max_refinements 2 \
  --Mapper.ba_global_frames_ratio 1.35 --Mapper.ba_global_points_ratio 1.35 \
  > $P/work/${SUB}_map.log 2>&1
echo "=== [$SUB] DONE $(date +%T) ==="
ls "$OUT/sparse"
