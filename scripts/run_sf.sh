#!/bin/bash
# SF downtown reconstruction. Phase A = May-15 session only (0023/0024/0025),
# all 3840x2880 like the site clip, so the site's camera params seed it.
# May-14 (0014/0015) needs its own camera model because it is FOV:MAX, and it
# bridges to May 15 only weakly, so it is a later phase. NB the deciding factor
# is the FOV tag, NOT the 2688x2016 resolution: the Aug-27 clips 0050-0060 are
# also 2688x2016 but FOV:Normal, i.e. the same field of view as these, just
# fewer pixels. Read `ffprobe -show_entries format_tags=comment`.
set -e
P=~/Desktop/fpv-splat
SET=sf
OUT=$P/colmap/sf; IMG=$P/frames/$SET
CLIPS="0023 0024 0025"
rm -rf "$OUT"; mkdir -p "$OUT/sparse" "$OUT/snapshots"

echo "=== FEATURES $(date +%T) ==="
colmap feature_extractor --database_path "$OUT/database.db" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "733,733,960,720,-0.1066,0.0077,0.0013,0.0004" \
  --FeatureExtraction.use_gpu 0 > $P/work/sf_feat.log 2>&1
grep -E "Elapsed" $P/work/sf_feat.log | tail -1

echo "=== SEQUENTIAL MATCH $(date +%T) ==="
colmap sequential_matcher --database_path "$OUT/database.db" --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 12 --SequentialMatching.quadratic_overlap 1 \
  > $P/work/sf_seq.log 2>&1
grep -E "Elapsed" $P/work/sf_seq.log | tail -1

echo "=== CROSS-CLIP MATCH $(date +%T) ==="
cd $P && $P/.venv/bin/python scripts/gen_cross_pairs2.py $SET $CLIPS --stride 8
colmap matches_importer --database_path "$OUT/database.db" \
  --match_list_path "$P/work/cross_pairs.txt" --match_type pairs \
  --FeatureMatching.use_gpu 0 > $P/work/sf_cross.log 2>&1
grep -E "Elapsed" $P/work/sf_cross.log | tail -1

echo "=== MAPPER $(date +%T) ==="
# Settings that made the site work: low init_min_tri_angle for forward motion,
# capped global BA refinements, and snapshots so a kill is never total loss.
nice -n 5 colmap mapper --database_path "$OUT/database.db" --image_path "$IMG" \
  --output_path "$OUT/sparse" \
  --Mapper.snapshot_path "$OUT/snapshots" --Mapper.snapshot_frames_freq 200 \
  --Mapper.init_min_tri_angle 3 --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_num_trials 500 --Mapper.min_model_size 40 \
  --Mapper.filter_min_tri_angle 1.0 --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.ba_global_max_refinements 2 \
  --Mapper.ba_global_frames_ratio 1.35 --Mapper.ba_global_points_ratio 1.35 \
  > $P/work/sf_map.log 2>&1
echo "=== SF DONE $(date +%T) ==="
ls "$OUT/sparse"
