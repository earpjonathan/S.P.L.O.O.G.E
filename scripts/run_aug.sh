#!/bin/bash
# Aug-27 hilltop reconstruction (clips 0050/0051/0052/0053/0055).
#
# This is a SEPARATE scene from colmap/both (May-17, clips 0026/0027), not an
# extension of it. The two sessions share no usable features: the dense probe
# found a peak of 76 SIFT inliers against a within-session control max of 57
# and ZERO pairs over 80 anywhere, versus 302 and 7 pairs over 80 for two clips
# of the same session. Every inlier in the best cross-session pair sat on the
# distant skyline -- 76 of 77 inside the thin horizon band, none on the ground
# being flown over -- which is far-field, so it constrains orientation and
# gives no triangulation baseline. The site itself was regraded in between
# (rocky in May, smooth graded dirt with a bulldozer in August).
#
# The clips ARE 2688x2016 like the rejected 0014/0015, but that is not what
# made those unusable: these are FOV:Normal, the same field of view as the
# 3840x2880 clips, just fewer pixels. Hence the same seeded camera params.
set -e
P=~/Desktop/fpv-splat
SET=aug
OUT=$P/colmap/aug; IMG=$P/frames/$SET
CLIPS="0050 0051 0052 0053 0055"
rm -rf "$OUT"; mkdir -p "$OUT/sparse" "$OUT/snapshots"

echo "=== FEATURES $(date +%T) ==="
colmap feature_extractor --database_path "$OUT/database.db" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "725,724,960,720,-0.107,0.008,0.001,0" \
  --FeatureExtraction.use_gpu 0 > $P/work/aug_feat.log 2>&1
grep -E "Elapsed" $P/work/aug_feat.log | tail -1

echo "=== SEQUENTIAL MATCH $(date +%T) ==="
colmap sequential_matcher --database_path "$OUT/database.db" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 12 --SequentialMatching.quadratic_overlap 1 \
  > $P/work/aug_seq.log 2>&1
grep -E "Elapsed" $P/work/aug_seq.log | tail -1

echo "=== CROSS-CLIP MATCH $(date +%T) ==="
# stride 10, not the 8 used for SF: five clips means ten cross-clip
# combinations instead of three, and the pair count grows with the square of
# the per-clip sample.
cd $P && $P/.venv/bin/python scripts/gen_cross_pairs2.py $SET $CLIPS --stride 10
colmap matches_importer --database_path "$OUT/database.db" \
  --match_list_path "$P/work/cross_pairs.txt" --match_type pairs \
  --FeatureMatching.use_gpu 0 > $P/work/aug_cross.log 2>&1
grep -E "Elapsed" $P/work/aug_cross.log | tail -1

echo "=== MAPPER $(date +%T) ==="
# Same hedge that got 99.3% on the site and 100% on SF: low init_min_tri_angle
# for forward motion, capped global BA refinements, snapshots so a kill is
# never a total loss.
nice -n 5 colmap mapper --database_path "$OUT/database.db" --image_path "$IMG" \
  --output_path "$OUT/sparse" \
  --Mapper.snapshot_path "$OUT/snapshots" --Mapper.snapshot_frames_freq 200 \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 40 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.ba_global_max_refinements 2 \
  --Mapper.ba_global_frames_ratio 1.35 --Mapper.ba_global_points_ratio 1.35 \
  > $P/work/aug_map.log 2>&1
echo "=== AUG DONE $(date +%T) ==="
ls "$OUT/sparse"
for m in "$OUT"/sparse/*/; do
  echo "  $m: $(colmap model_analyzer --path "$m" 2>&1 | grep -E 'Cameras|Images|Points|observations|length' | tr '\n' ' ')"
done
