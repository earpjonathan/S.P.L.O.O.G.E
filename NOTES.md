# fpv-splat — FPV footage → SfM → 3D Gaussian Splatting

Standalone project dir (deliberately **not** inside any git repo — holds multi-GB
video and model files).

## Machine / toolchain

| tool | version | notes |
|---|---|---|
| Apple M5 Pro (arm64) | macOS 25.5.0 | 15 cores |
| ffmpeg | 9.0.1_1 | see *x265 trap* below |
| COLMAP | 4.1.1 | **without CUDA** — CPU SIFT |
| Gyroflow | 1.6.3 | `/Applications/Gyroflow.app/Contents/MacOS/gyroflow` has a CLI |
| Python venv | 3.14.7 | `.venv/` — numpy, scipy, opencv, matplotlib, cbor2 |

### x265 trap (cost ~10 min)
`brew install colmap` left Homebrew with x265 4.3 *linked but with an empty
`lib/`*, while ffmpeg 9.0.1 was built against `libx265.216.dylib` (x265 4.2).
Both `ffmpeg` and `colmap` died at dyld load. `brew reinstall x265` fixed the
empty dir but installed soname **217**, breaking ffmpeg for real.
Fix: `brew reinstall ffmpeg` → picks up revision `9.0.1_1` built against 4.3.
If either binary ever fails with `Library not loaded: ...libx265.NNN.dylib`,
reinstall ffmpeg, not x265.

### COLMAP 4.1.1 option renames
Options moved from `--SiftExtraction.*` to `--FeatureExtraction.*` for
`use_gpu`, `num_threads`, `max_image_size`. `--SiftExtraction.*` still exists
for the actual SIFT tuning knobs. COLMAP 4.x also ships **ALIKED** learned
features (`--FeatureExtraction.type ALIKED`) — a fallback worth trying if SIFT
struggles on dim/blurry footage.

## Source clip

`DJI_20260821201604_0049_D.MP4` (copied to `clips/0049.MP4`, 3.59 GB)

| property | value |
|---|---|
| codec | H.264 High, 100.5 Mbit/s |
| resolution | **2688 × 2016 (4:3)** |
| frame rate | 50.00 fps constant |
| duration | 272.06 s (13 603 frames) |
| encoder tag | `DJI O4P` |
| **comment tag** | **`EIS:OFF;FOV:Normal;`** |

### Gyro telemetry — AVAILABLE
The precondition given was "EIS off **and** FOV Wide". This clip is EIS:OFF but
FOV:**Normal** — yet gyro extraction works anyway. FOV:Wide governs which *lens
profile* applies, not whether DJI writes telemetry.

Stream 1 (`DJI meta`) holds 15 MB of `dvtm_O4P.proto` protobuf. Gyroflow parses
it and reports `IMU duration 272058.99 ms` vs video 272060 ms — full coverage.

```bash
/Applications/Gyroflow.app/Contents/MacOS/gyroflow clips/0049.MP4 \
  --export-metadata "3:gyro/camera_0049.json"
```

Yields **13 603 samples = exactly one per video frame**, each with `org_quat`
(camera orientation, `[w,x,y,z]`), `org_euler`, `stab_quat`, `timestamp_ms`.
`org_gyro` and `org_acc` are **all zeros** — DJI exposes only the fused
orientation, not raw rates. Orientation only; no position, no GPS.

### Lens / camera model
Gyroflow's bundled DB (12 409 profiles) has 6 `DJI_O4` entries — all **O4
Lite / Air Unit** with third-party Flywoo lenses at 3840×2880 or 3840×2160
@59.94. **None matches an O4 Pro at 2688×2016@50 FOV:Normal.** No off-the-shelf
calibration exists for this mode.

Focal length was therefore derived **empirically from the gyro**: during
rotation, image displacement ≈ `f_px × Δθ`. Regressing measured Lucas-Kanade
flow against gyro rotation over the 337° spin at t≈260 s (`scripts/est_focal.py`):

```
focal = 965.4 px @ 2688 wide      R² = 0.967  (359/392 inlier pairs)
=> HFOV 108.6°   VFOV 92.5°   DFOV 120.2°   f/W = 0.359
```

Used as the COLMAP prior with the **OPENCV** model (fx,fy,cx,cy,k1,k2,p1,p2),
single shared camera, refined during mapping. Powerlines visibly bow at frame
edges, so distortion is real and uncorrected — 120° DFOV is within OPENCV's
range, with `OPENCV_FISHEYE` as fallback.

## Frame extraction rate

Not a fixed default — measured. Lucas-Kanade flow on native-50fps bursts at
t = 50/100/150/200/230 s (`scripts/flow_bursts.py`):

```
median 9.4 px/frame @50fps   (p75 13.9, p95 21.2)  in full-res px
```

Small, because motion is **forward-dominated** — radial expansion about the
focus of expansion, not lateral parallax. Decimation table:

| keep every | eff. fps | median disp | p95 disp |
|---|---|---|---|
| 2 | 25.0 | 18.8 px | 42.4 px |
| **5** | **10.0** | **47.0 px** | **105.9 px** |
| 10 | 5.0 | 94.1 px | 211.8 px |

**Chose 10 fps (every 5th frame).** 47 px median displacement is a comfortable
SIFT matching baseline while still giving real triangulation baseline; p95 of
106 px stays well inside SIFT's range even with dusk motion blur. 5 fps was
rejected as too risky given measured blur; 25 fps wastes compute on
near-duplicate views.

## Segment choice

Sharpness (Laplacian variance) and brightness scanned over the whole clip at
2 fps (`scripts/survey_quality.py`). The clip **degrades monotonically** — it
was shot into dusk:

```
t=0-40s    sharpness 312   brightness 85
t=100-140  sharpness  92   brightness 60
t=200-240  sharpness  57   brightness 47
```

Chose **t = 0–40 s → 400 frames**. Caveat: part of that sharpness score is
extreme-foreground gravel during the ground/launch phase, not scene texture.

### Frame-rate validation (important correction)
The 10 fps figure was first derived from cruise-phase bursts at t=50–230 s, but
the chosen segment is t=0–40 s, which contains the launch. Re-measured **inside**
the segment at native 50 fps:

```
t= 5s  11.6 px/frame   t=12s  12.9   t=20s  17.9   t=28s  12.4   t=35s   5.3
```

Launch itself (t≈1.0–2.5 s) peaks at **135–246 px/frame** — a punch-out.

A Lucas-Kanade diagnostic across the 10 fps frames returns *no solution* between
t=3 s and t=11.5 s. **This is an artefact of the diagnostic, not the footage** —
LK does a local search (winSize 21, 4 pyramid levels ≈ ±168 px) and a 5× frame
jump exceeds it. The same span tracks 533–761 features cleanly at native 50 fps,
so the texture is there.

SIFT matches globally and is far more tolerant of large displacement. Confirmed
against the actual match graph:

```
verified pairs                 2 890
consecutive (n,n+1) verified   399 / 399      <- unbroken chain
  inliers  median 864   min 183   p10 430
weak links (<100 inliers)      0
match-graph degree             median 15, min 7
```

**10 fps stands.** Never infer SfM matchability from optical-flow failure.

---

# PHASE 1 RESULTS

## Pipeline actually run

```bash
colmap feature_extractor --ImageReader.camera_model OPENCV \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "965.4,965.4,1344.0,1008.0,0,0,0,0" \
  --FeatureExtraction.use_gpu 0                       # 2.5 min, ~14.5k feat/image
colmap sequential_matcher --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 10 --SequentialMatching.quadratic_overlap 1
                                                      # 1.9 min, 2890 verified pairs
colmap mapper ...                                     # ~6 min -> 4 models (!)
colmap model_merger --input_path1 sparse/2 --input_path2 sparse/3
colmap bundle_adjuster --BundleAdjustment.refine_principal_point 0
```

## The mapper fragmented — and why

Default mapper produced **four** models, not one:

| model | images | frame range | breaks | points | reproj |
|---|---|---|---|---|---|
| 0 | 10 | 36–52 | 1 | 2 306 | 1.139 |
| 1 | 10 | 233–302 | 1 | 862 | 0.918 |
| **2** | **302** | **1–302** | **0** | 26 135 | 0.735 |
| 3 | 118 | 282–400 | 1 | 17 450 | 0.970 |

Union = **400/400 = 100 %**. Nothing failed to register — models 2 and 3
**overlap on frames 282–302**, so COLMAP simply never merged them.

Root cause is the initialisation gate, not the footage:
`--Mapper.init_min_tri_angle` defaults to **16°**, and forward flight down a
road produces triangulation angles far below that near the focus of expansion.
`--Mapper.init_max_forward_motion` (0.95) rejects forward-dominated pairs for
the same reason. Models 0 and 1 are junk stubs at `min_model_size` (10).

`model_merger` merged 2+3 cleanly (21 shared frames), then `bundle_adjuster`
converged: **initial cost 7.696 px → final 0.667 px**, 92 iterations, 40 s.

## Metrics vs the stated criteria — merged + bundle-adjusted

| criterion | target | result | |
|---|---|---|---|
| frames registered | >85 % | **400/400 = 100 %** | PASS |
| mean reprojection error | low | **0.85 px** (99.5 % of points) | PASS |
| sparse 3D points | — | **42 275** | PASS |
| trajectory continuous | by eye | **0 breaks, 0 step outliers** | PASS |
| gyro rotation cross-check | — | **r = 0.952, RMS 0.42°** | PASS |

Reprojection caveat: the raw mean is meaningless (1.9e150) because **6 points**
out of 42 275 are degenerate near-infinite triangulations. Filtering
`err<4px & dist<1000 camera-steps` keeps **99.5 %** of points at mean 0.846 px,
median 0.723 px, mean track length 7.27. Those 6 points should just be dropped.

### Camera self-calibration vs the gyro-derived prior
```
prior  (gyro/flow)  fx = 965.4   HFOV 108.6°  DFOV 120.2°
COLMAP refined      fx = 1046.7  HFOV 104.2°  DFOV 116.2°
                    k1 = -0.106  k2 = 0.0073  (barrel, as expected)
```
8 % apart — good agreement given the prior is a single global average taken
across a distorted field. The OPENCV model was the right choice; no need for
OPENCV_FISHEYE at this FOV.

### Rotation cross-check (the real validation)
COLMAP median inter-frame rotation **1.347°** vs gyro **1.344°** over 399
consecutive pairs, r = 0.952, RMS difference 0.42°. An IMU and a purely
image-based pipeline agreeing this closely means the recovered rotations are
genuinely correct, not a plausible-looking drift. Note this validates
**rotation only** — the DJI data carries no position, so translation and scale
drift are unchecked.

## The fix: retuned mapper (this is the result to use)

```bash
colmap mapper --database_path .../database.db --image_path frames/seg000 \
  --output_path colmap/seg000/sparse_tuned \
  --Mapper.init_min_tri_angle 3 \        # was 16 — the actual blocker
  --Mapper.init_max_forward_motion 1.0 \ # was 0.95 — stop rejecting forward pairs
  --Mapper.init_num_trials 500 \
  --Mapper.init_min_num_inliers 80 \
  --Mapper.min_model_size 25 \           # was 10 — suppress junk stubs
  --Mapper.filter_min_tri_angle 1.0 \
  --Mapper.ba_local_min_tri_angle 3 \
  --Mapper.abs_pose_min_inlier_ratio 0.2
```
6.5 min. **One model, all 400 frames, no merge step needed.**

| metric | default mapper | **tuned mapper** |
|---|---|---|
| models | 4 (fragmented) | **1** (+1 8-frame stub) |
| largest model | 302 / 400 = 75.5 % | **400 / 400 = 100 %** |
| sparse points | 26 135 | **45 165** |
| mean reproj error | 0.735 (largest model only) | **0.826 px** (clean, no outlier blow-up) |
| breaks in sequence | 0 | **0** |
| step outliers | 0 | **0** |
| gyro rotation r | 0.940 | **0.951** (RMS 0.43°) |

`reports/seg000_1_trajectory.png`, `reports/seg000_1_rotation_check.png`,
`reports/seg000_cloud.png`.

**Takeaway:** COLMAP's defaults are tuned for photogrammetry-style capture with
wide baselines. Forward-flying FPV video needs `init_min_tri_angle` dropped and
`init_max_forward_motion` opened up, or it silently fragments into sub-models
that look like a registration failure but aren't.

---

# VERDICT: PASS (SfM) — but the clip is wrong for the goal

Every stated Phase 1 criterion is met, comfortably. COLMAP recovers a clean,
continuous, gyro-validated camera trajectory from this footage.

**However** — the geometry it recovers is a *corridor*, not a *location*:

```
camera path extent   34.8  x  2.5  x  0.9     (path : lateral spread = 14 : 1)
PCA singular values  1.00  x  0.062 x 0.022
```

The drone flies down a road behind a car. Every surface is seen from a narrow
range of angles for a few seconds, then left behind. `reports/seg000_cloud.png`
shows the consequence: a thin ribbon of well-constrained geometry hugging the
flight line, plus a large fan of poorly-conditioned far-field points ahead of
the drone, triangulated along near-parallel rays.

Three further problems specific to this clip:
1. **The chased car is in ~every frame** (~4–6 % of frame area, near-centre).
   Because it moves *with* the drone it is near-stationary in image space, so
   it behaves like a point at infinity rather than an obvious outlier. It would
   ghost badly in a splat.
2. **Shot into dusk.** Sharpness falls 312 → 57 and brightness 85 → 47 across
   the clip. Only the first ~40 s is good, which is why that segment was used.
3. **Forward-dominated motion** gives weak triangulation — the root cause of
   both the fragmentation and the far-field point fan.

A splat trained on this would look acceptable from on/near the original flight
line and fall apart as soon as the viewer moves off it.

## Better source clips on the card

All 49 clips are `EIS:OFF`, so gyro telemetry is available on every one.
Clips **0023–0027 are 3840×2880** (4K 4:3) — and that resolution exactly matches
the one usable O4 lens profile in Gyroflow's database.

Camera-axis dispersion (`scripts/orbitness2.py`, `1 - |mean unit vector|`;
0 = never reorients, 1 = sweeps all directions) separates orbits from corridors:

| clip | dur | res | dispersion | content |
|---|---|---|---|---|
| **0026** | 145 s | 3840×2880 | **0.954** | **hilltop construction site — bounded, matte, orbited** |
| **0027** | 144 s | 3840×2880 | **0.953** | same site, same session (4 min later) |
| 0025 | 200 s | 3840×2880 | 0.954 | downtown SF, Salesforce Tower orbits |
| 0023 | 241 s | 3840×2880 | 0.640 | partial coverage |
| 0049 | 272 s | 2688×2016 | **0.204** | car chase — **corridor** (this clip) |

**Recommended: 0026 + 0027 together** — 288 s of genuinely orbital coverage of a
single bounded location, midday sun, matte dirt with dense non-repetitive
texture (tire tracks, gravel, rock piles), static subjects, no moving object
dominating frame. Almost the ideal 3DGS capture.

0025 (SF) is more spectacular but a harder case: glass curtain walls are
strongly repetitive (false matches) and specular (view-dependent, bad for both
SfM and splatting).

Note 4K 4:3 shows much stronger barrel distortion than 2688×2016 — the horizon
visibly bows. `OPENCV_FISHEYE` should be tried alongside `OPENCV` for those.

## Phase 2 readiness
Not started — awaiting explicit go-ahead per the brief. Rust/cargo and Brush
have deliberately **not** been installed yet.

---

# CLIPS 0026 + 0027 (construction site) — Phase 1 redo

Both `3840×2880 @ 50 fps`, ~144 s each, recorded 4 min apart (11:38:35 /
11:42:29) — same session, same hilltop site. Both `EIS:OFF;FOV:Normal;`.

## Camera model: the pilot overruled the analysis

Rotation here is ~5× faster than the chase clip (median 51–60°/s, peaks >900°/s).

The gyro-vs-flow focal fit that worked so well on 0049 fit **badly** here
(R² 0.648 vs 0.967). A radius-resolved profile suggested strong fisheye:

```
 r/halfW   0.06  0.19  0.31  0.44  0.56  0.69  0.81  0.94  1.06  1.19
 slope     1222  1159  1164  1343  1492  1597  1664  1617  1602  1336   px/rad
 centre->edge ratio 1.09    pinhole predicts 4.48    equidistant predicts 1.00
```

So a 300-frame pilot was run with **both** camera models before committing:

| model | registered | models | points | reproj |
|---|---|---|---|---|
| **OPENCV** | **295/300 = 98.3 %** | **1** | 144 567 | 0.788 |
| OPENCV_FISHEYE | 156/300 = 52.0 % | 4 (fragmented) | 58 601 | 0.653 |

**OPENCV wins decisively** — and converges to `k1 = -0.107`, essentially the
same as clip 0049's `k1 = -0.104`, i.e. DJI applies the same dewarp in both
modes. The radius profile was a **flawed diagnostic**: displacement under
rotation also depends on the angle between each feature and the rotation axis,
which the single-slope fit does not model, flattening the measured curve.
Lesson: pilot the camera model, don't infer it from a flow profile.

## Adaptive, gyro-driven frame selection

Uniform decimation cannot work here — fast rotation demands dense sampling, but
289 s at 16.7 fps is 4 800 frames. Uniform sampling also leaves a brutal tail
(at 10 fps the p99 inter-frame rotation is 49–66°).

Instead frames are kept whenever **accumulated gyro rotation crosses 8°**
(min 2, max 15 frames apart) — `scripts/select_frames.py`:

| clip | frames | median rot | p90 | p99 | median px @1920 |
|---|---|---|---|---|---|
| 0026 | 1 156 | 8.81° | 11.26° | 29.14° | 108 |
| 0027 | 1 312 | 9.00° | 13.08° | 35.90° | 110 |

**2 468 frames**, near-constant baseline, tail largely gone. This is the gyro
earning its keep beyond the Phase 1 sanity check.

Cross-clip linking: sequential matching alone can never connect 0026 to 0027.
No vocab tree ships with COLMAP, so instead an explicit candidate pair list
(every 8th frame of each clip, crossed, plus intra-clip long-range pairs) is
fed through `colmap matches_importer` — `scripts/gen_cross_pairs.py`.

---

# PHASE 2 — Brush on Apple Silicon

## Build
`brew install rust` → rustc/cargo 1.98.0 (Brush needs edition 2024 ≥ 1.85).
`cargo build --release -p brush-cli` → **6 min 14 s**, 176 MB binary.

## Metal confirmed
Brush does not log its wgpu adapter, so it was confirmed three ways:
- `otool -L brush-cli` links **`Metal.framework`** and no Vulkan/OpenGL.
- 1 000 iters at full 1920 px in **83 s (12 it/s)**, peak RSS **3.77 GB**.
- Of 83 s wall only 50 s was user CPU — the CPU is *waiting on the GPU*.
  Pure-CPU compute would show user time far exceeding wall time.

**Verdict: Brush is viable on this machine. The GTX 1660 Super fallback is not
needed.** 30 k iterations projects to ~40–70 min.

## Trial results (295-image pilot reconstruction)
5 000 iterations at 1600 px in ~7 min, eval split every 20th image.
`work/cmp_003597.jpg` compares a **held-out** eval view against ground truth:
terrain shape, tree line, distant houses and colour all reconstruct correctly;
soft with some dark floaters near trees, as expected this early.

Stability note: the 600-iter export contained **279 non-finite floats across
93 splats** (0.002 %). Small, but worth re-checking on the full run — the
converter drops them.

## Viewer
`viewer/` = antimatter15/splat, with two documented patches:
1. `?url=` now resolves against the page's own location instead of a hardcoded
   HuggingFace base, so it works self-hosted (and later when embedded).
2. Canvas sizing falls back to the canvas box when `innerWidth/innerHeight`
   report 0 — a 0 there makes the projection matrix NaN.

`scripts/ply2splat.py` — vectorised PLY→.splat (0.14 s vs minutes for the
bundled `convert.py`, which loops per splat in Python). It also drops
non-finite splats and **normalises the scene into the viewer's frame**: it reads
Brush's `comment Vertical axis:` from the PLY header, rotates that axis to
(0,-1,0), centres on the median, and scales so the p90 radius is 3 units.
Without this the scene sits far outside the viewer's default camera.

### Viewer verified rendering

The Claude Browser pane runs with `visibilityState: "hidden"` and (initially)
`innerWidth/innerHeight = 0`. That combination pauses `requestAnimationFrame`
after a single frame and makes the projection matrix NaN — the viewer showed
only its loading spinner, and did so **on its own reference scene too**, so it
was never a fault in the splat or the converter.

Fixed/worked around in three steps:
1. `resize_window` to give the pane a real 1280x720 viewport.
2. The zero-viewport guard above (a real bug worth keeping for embeds).
3. `verify.html` — a copy of `index.html` with `requestAnimationFrame`
   polyfilled onto `setTimeout`, **for verification only**. `index.html` ships
   clean; never use the polyfill in production, it spins when hidden.

With those, the viewer renders the site correctly from a real capture pose.
Two debugging traps worth remembering:
- `index.html?cb=…` does **not** cache-bust `main.js`. Bump a query on the
  `<script src>` itself or you debug a stale file (this cost a long detour).
- With 1.2 M splats the first paint takes several seconds on the polyfilled
  loop — "still spinner" is not the same as "broken".

### Capture poses as viewpoints
`ply2splat.py --colmap <sparse_dir>` now also:
- normalises using **camera positions** rather than the splat median (the
  median is dragged around by sky/background blobs and drops the default
  camera inside the scene), and
- emits a `cameras.json` of the real capture poses, transformed through the
  same level/centre/scale.

The viewer was patched to auto-load `cameras.json` next to the splat, so it
opens on a real drone viewpoint and keys `0-9` step through captures. Its
resize handler now scales `fx/fy` from source-image pixels to the viewport
(uniformly, off height) — previously the field of view changed with window size.

Serve locally with:
`python3 -m http.server 8777 --directory ~/Desktop/fpv-splat/viewer`
then open `http://localhost:8777/index.html?url=scene.splat`

### Brush stability across the trial
| export | splats | non-finite splats |
|---|---|---|
| 600 iters | 211 366 | 93 (0.044 %) |
| 5 000 iters | 2 636 477 | **0** |

The early NaNs are transient and clean themselves up — Brush is stable.
Densification is aggressive though (2.6 M splats by 5 k iters, 622 MB PLY), so
web exports use `--max-splats` (ranked by opacity x projected area);
1.2 M splats = 36.6 MB.

---

# SITE RECONSTRUCTION — result

## The combined 0026+0027 run overran
The 2 468-frame mapper ran **>4 hours** without finishing. Diagnosis while it
ran: process state `RN`, CPU time accumulating (~4.5 cores), RSS flat at
7.1 GB for long stretches = stuck inside **global bundle adjustment**, not
registering. It never fragmented (no sub-model dirs), so it was still growing
one model — but with `ba_global_max_refinements 5` on 2 400 images the global
BA cost is brutal.

**Two mistakes worth recording:**
1. `run_site.sh` pipes the mapper through `grep … | tail -3`. `tail` buffers to
   EOF, so there was **no incremental progress visible at all** for four hours.
2. `--Mapper.snapshot_path` was not set, so killing the run would have salvaged
   **nothing** — COLMAP only writes models at the end.

## The hedge run — this is the one that shipped
Rather than kill 4 hours of work, a second mapper was started on a **copy of
the same database** (features/matches reused, nothing recomputed), restricted
to clip 0026 via `--Mapper.image_list_path`, `nice`d so it would not starve the
first, and — critically — **with snapshots enabled**:

```
--Mapper.image_list_path work/list_0026.txt
--Mapper.snapshot_path colmap/hedge26/snapshots --Mapper.snapshot_frames_freq 200
--Mapper.ba_global_max_refinements 2       # was 5
--Mapper.ba_global_frames_ratio 1.35       # was 1.1 -> global BA far less often
--Mapper.ba_global_points_ratio 1.35
```

Finished in **50.6 minutes**, and the snapshots gave live progress the main run
never did (202 -> 402 -> 602 -> 802 -> 1002 images, reproj holding 0.84-0.88).

## Metrics vs the Phase 1 criteria — clip 0026

| criterion | target | result | |
|---|---|---|---|
| frames registered | >85 % | **1 144 / 1 156 = 99.0 %** | PASS |
| mean reprojection error | low | **0.876 px** (100 % of points under 4 px) | PASS |
| sparse 3D points | — | **620 600** | PASS |
| trajectory continuous | by eye | **no gaps**, loops over the site | PASS |
| gyro rotation cross-check | — | **r = 0.9989**, RMS **0.288°** on 98.3 % of pairs | PASS |

Camera self-calibrated to `fx = 727.9, k1 = -0.108` — consistent with the
300-frame pilot (733 / -0.107), so the OPENCV choice held up at full scale.

Gyro residuals: median **0.109°**, p90 0.468°. 19 pairs of 1 143 (1.66 %)
disagree by >2°, 6 by >10° — a small set of genuinely mis-posed frames. The
headline r=0.9989 / RMS 0.288° is quoted **on the 98.3 % that agree**; the raw
RMS over all pairs is 4.76°, inflated by those few outliers.

## This is a location, not a corridor
```
                 PCA singular values        interpretation
chase clip 0049  1 : 0.062 : 0.022          14:1 ribbon — a corridor
site  clip 0026  1 : 0.717 : 0.150          compact 2D area with vertical relief
```
`reports/site_0_cloud.png`: a dense circular core with the flight path looping
*inside* it, on a thin ground plane, ringed by a weakly-conditioned halo of
distant city. Exactly the coverage 3DGS needs.

### Correction: what the gyro cross-check actually validated

I earlier reported the 0026 gyro check as "r = 0.9989, RMS 0.288 deg on the
98.3 % of pairs within 2 deg" and presented it as validating recovered camera
*orientation*. That overstated it. Re-deriving it (`scripts/gyro_check.py`):

- It was a **magnitude-only** check: it compared *how much* the camera rotated
  between consecutive frames, COLMAP vs gyro. It never compared rotation
  **axes**. Relative-rotation magnitude is invariant to the camera<-imu
  alignment, which is why it needed no alignment solve.
- The `r = 0.9989` was computed **on the within-2-degree subset only**.
  Correlating after discarding the disagreements inflates r. Honest
  full-sample figure: **r = 0.8296**, RMS 4.763 deg.

What survives, and it is not trivial (`median |dtheta|`, 1143 pairs):

| pairing | median abs diff | r |
|---|---|---|
| true | **0.109 deg** | 0.8296 |
| shuffled control | 1.250 deg | 0.0038 |

Gyro inter-frame angles span 0.05-177.85 deg (std 7.2), so this is a real
match, not two tight clusters coinciding.

**Axis agreement remains unvalidated.** Every attempt to compare full rotations
needs the fixed camera<-imu alignment A, and every fit of A here is
ill-conditioned or wrong:

| method | in-result | why it is not trustworthy |
|---|---|---|
| Kabsch on inter-frame rotation axes | median 4.23 deg | axes near-collinear, SVD **1 : 0.089 : 0.019** - the clip is an orbit, so nearly every inter-frame rotation shares an axis |
| two-sided Procrustes on absolute orientations | median 21.6 deg | single global A,B cannot absorb gyro yaw drift |
| same, fitted per 60-frame window | median 11.6 deg | drift now absorbed, residual persists - so the model, not drift, is wrong |

Do not read those numbers as pose error. **A 10 deg rotation error at
f = 727.9 px would displace features by ~127 px; COLMAP's mean reprojection
error is 0.876 px.** The two cannot both be true.

The pose validation that actually holds is photometric, and it is stronger than
the gyro check ever was: **held-out** eval frames render into near-exact
correspondence with ground truth (`reports/eval_5000_best.jpg` - trucks, tree,
dirt mound, city skyline all land in place at only 5 k iterations).

Lesson: a rotation-magnitude check is alignment-free and therefore easy, but it
is a much weaker claim than "orientation agrees". State which one you ran. And
never quote a correlation computed on the subset that excludes the outliers.

### Viewer: opening view (`?cam=N`)

Camera 0 is just the first frame of the flight. On clip 0026 that frame points
at sky, so the viewer opened on a blue void and looked broken. Two changes:

- `scripts/pick_view.py <splat> <cameras.json>` scores every capture pose by how
  much of the scene it actually frames (fraction of splats inside the frustum +
  share of total opacity, so a sky-filled view scores low). On the 10 k preview
  it picks **cam 16** (`0026_002216`): 68.7 % of splats in frustum, 74.2 % of
  opacity mass. That independently matched the frame that looked best by eye.
- `main.js` gained a third patch: `?cam=N` selects the opening pose. Two traps
  hit while writing it, both now fixed:
  - `currentCameraIndex` is declared with `let` *below* the cameras.json block,
    so assigning it there threw a temporal-dead-zone `ReferenceError`.
    Declaration hoisted.
  - `carousel` defaults to true and animates the camera away from the start
    pose within a second, so `?cam=` appeared to do nothing. An explicit
    `?cam=` now switches the carousel off and sets `viewMatrix` directly
    (a `#hash` viewpoint still wins, being more specific).

Remember `index.html?cb=` does **not** cache-bust `main.js` - the script tag
carries its own `?v=`, so bump that after editing or the browser serves the old
file. This wasted a long debugging session once already.

## PHASE 2 RESULT - site reconstruction (clip 0026)

Brush, Metal backend, 30 000 iterations on the 1 144 registered frames of the
hedge reconstruction. **16:59:50 -> 19:54:39 = 2 h 54 m 49 s.**

### Held-out eval (every 40th image, 29 frames, never trained on)

| iter | mean PSNR | median | min | max | splats |
|---|---|---|---|---|---|
| 5 000  | 21.27 | 22.34 | 11.38 | 24.57 | - |
| 10 000 | 22.01 | 22.57 | 14.43 | 25.03 | 4 907 525 |
| 15 000 | 22.34 | 23.20 | 13.43 | 26.20 | - |
| 20 000 | 22.29 | 23.01 | 15.68 | 26.27 | **6 000 000 (cap)** |
| 25 000 | 22.58 | 22.88 | 16.48 | 26.56 | 6 000 000 |
| 30 000 | **22.44** | **22.96** | **16.39** | **26.30** | 6 000 000 |

The mean plateaus after ~15 k. What keeps improving is the **minimum**
(11.4 -> 16.4 dB): late training cleans up outlier views rather than sharpening
the average. Anyone reading only the mean would conclude training stalled at
15 k and stop early; the distribution says otherwise.

**Densification, not iteration count, paced this run.** Per-5k-block wall time
was 14m41s, 27m02s, 34m01s, then 36m02s and 36m01s once the 6 M splat cap was
reached - within one second of each other. The cap is now the binding
constraint on quality, not iterations: 6 M splats over 1 144 images at 1920 px
is not generous. Raise `--max-splats` before raising `--total-train-iters`.

### Stability

The 10 k export contained 1 561 non-finite splats (0.03 %); the 30 k export
contained **zero**. Same pattern as the trial (93 non-finite at 600 iters, 0 at
5 000). Non-finite splats are an early-training transient, not a Metal defect.

### Deliverables

| file | splats | size | use |
|---|---|---|---|
| `viewer/site.splat` | 3 000 000 | 92 MB | local flying, 94-108 fps |
| `viewer/site_web.splat` | 1 500 000 | 46 MB | portfolio embed |
| `viewer/cameras.json` | 58 poses | 23 KB | viewpoint presets (stride 20) |

```
python3 -m http.server 8777 --directory ~/Desktop/fpv-splat/viewer
open "http://localhost:8777/index.html?url=site.splat&cam=16"
```

Watch out: passing `--colmap` without `--cameras-stride` rewrites
`cameras.json` at stride 1 (1 144 entries), which makes `+`/`-` stepping
useless. The second (web) export did exactly that and had to be redone.

### The combined 0026+0027 mapper: killed at 9 h 38 m, nothing salvageable

Ran 12:59 -> 22:38 (**9 h 38 m, 1 605 min CPU**) on 2 468 images with 47 586
match pairs. It wrote **zero** models. Killed on request; `colmap/site/sparse/`
was empty then and is empty now, so nothing was lost by killing it and nothing
would have been gained by waiting.

Three mistakes made this unrecoverable rather than merely slow:

1. **No `--Mapper.snapshot_path`.** With snapshots, killing at any point yields
   a usable partial model. Without them, 9 h of CPU produces nothing. The
   hedge run had them and that is precisely why its progress was visible.
2. **`colmap mapper ... | grep ... | tail -3`** - `tail` buffers to EOF, so the
   log stayed empty for the entire run. There was never any way to tell
   "working" from "wedged". Never put `tail` in a live progress pipeline; drop
   it and let `grep --line-buffered` through.
3. **Default global-BA settings at 2 468 images.** The hedge used
   `--Mapper.ba_global_max_refinements 2` and
   `ba_global_frames_ratio/points_ratio 1.35`, and finished 1 156 images in
   **50.6 min**. The combined run kept the defaults and never emerged from
   global BA.

To fold 0027 in later, re-run with the hedge's BA settings *and* snapshots. The
features and matches are already in `colmap/site/database.db`, so nothing needs
recomputing - that database is the salvageable part, and it survived.

## Sky floaters and the flight path

### The "clouds" are not clouds

The footage has clear sky. Those blobs are **sky Gaussians**: sky has no
parallax, so its depth is unconstrained - the optimiser can place a
sky-coloured splat anywhere along the ray and still match every training
image. Many land low over the terrain and read as badly-formed clouds.
They are large, faint and blue-shifted:

| | near real geometry | floaters |
|---|---|---|
| median largest-axis scale | 0.0066 | **0.1686** (25x) |
| blueness (B - mean(R,G)) | -0.035 | **+0.108** |

### The filter that did NOT work

First attempt dropped splats far from any COLMAP-triangulated point and large.
It punched **black holes in the ground**. Cause: textureless dirt yields few
SIFT features, so *real* ground splats there are far from any triangulated
point, and being smooth they are also large - they matched both criteria.
**14.6 % of what it dropped was terrain-level geometry.** Distance-to-points is
a texture measure, not a floater measure.

### The up-axis was inverted (this cost the first attempt)

Brush's `comment Vertical axis:` points **DOWN** (gravity). `ply2splat` maps it
to `(0,-1,0)`, so after levelling **up is +Y**, not -Y. Confirmed two ways:

- 66 % of COLMAP points lie *below* the cameras (a drone flies above terrain);
- only 22 % of frames have `forward.Y > 0`, yet the drone plainly looks down.

With the sign wrong, `--max-height 8` was cutting things 8 units *underground*
instead of sky, and the terrain check put the cameras *below* ground.
**Sanity check any vertical convention by asserting the cameras are above the
terrain.** It is one line and it catches an inverted axis immediately.

### The filter that works: height above local terrain

`--strip-floaters` builds a terrain height field (96x96 grid, 60th percentile
of COLMAP point heights per cell) and drops splats by AGL. Real surface splats
sit at terrain height whatever their texture; floaters do not. The distribution
is cleanly bimodal - p95 of splat AGL is **0.09**, p99 is **4.18**.

The grid extent must be the **flight area**, not the full point cloud: the
cloud includes distant city and hills, which stretched the grid until each cell
spanned a whole hillside.

Outside the flight area there is no terrain estimate, so only a global ceiling
applies (`--sky-margin`, default highest camera + 2.0). Out there sky splats
reach height **139.8** against a highest camera of **4.40**.

Result: **3.42 %** dropped, terrain fully intact.

### Flight path

`--path` bakes the camera track in as splats, cyan at the start to magenta at
the end. Drawn as Gaussians rather than a GL line **on purpose**: this viewer
has no depth buffer, it sorts and alpha-blends, so a line primitive would draw
straight through the terrain. Splats join the same sort and occlude correctly.
Appended after `--max-splats` so they are never culled. 20 426 markers over
1 144 poses.

### Files

| file | splats | size | |
|---|---|---|---|
| `site_clean.splat` | 3 000 000 | 92 MB | floaters removed |
| `site_path.splat`  | 3 020 426 | 92 MB | floaters removed + flight path |
| `site_web.splat`   | 1 500 000 | 46 MB | floaters removed, for embedding |
| `site.splat`       | 3 000 000 | 92 MB | unfiltered original, for comparison |

```
open "http://localhost:8777/index.html?url=site_path.splat&cam=16"
```

## Can other flights at this location be added?

Short answer: **clip 0027 yes, everything else no** - and the blocker is not
lighting, it is that the other sessions do not overlap at all.

### Measured cross-clip matchability

18 frames spread over each clip, all-pairs SIFT + ratio test + RANSAC
(`scripts/xclip_match2.py`, 324 pairs per clip). A within-clip control fixes
the noise floor: 0026 against its own consecutive samples gives **median 12**,
max 92. Median ~12 is therefore *no match*, not a weak one.

| pair | max | p90 | >=40 | >=80 | verdict |
|---|---|---|---|---|---|
| 0026-0027 | **198** | 24 | 19 | 10 | strong overlap |
| 0026-0008 | 40 | 16 | 1 | 0 | weak (noise floor) |
| 0026-0039 | 35 | 16 | 0 | 0 | no usable overlap |
| 0026-0040 | 31 | 15 | 0 | 0 | no usable overlap |

A first version of this test used only 5 frames per clip and made 0026-0027
look "marginal" (best 79). That was under-powered - with sparse sampling most
frame pairs simply do not share a viewpoint. **Always include a within-clip
control** so the noise floor is visible, and read the tail, not the median.

### Session dates (from the filenames)

| clip | when | what |
|---|---|---|
| 0008 | 2026-04-26 16:51 | same hill, green spring grass, before grading |
| **0026** | **2026-05-17 11:38** | the site, graded dirt - used |
| **0027** | **2026-05-17 11:42** | same site, 4 min later - unused, ready |
| 0039 | 2026-08-12 19:36 | golden hour, low sun and flare |
| 0040 | 2026-08-12 19:40 | landscaped: paved paths and plantings |
| 0025 | 2026-05-15 11:05 | downtown, not this location at all |

### Why different sessions would harm rather than help

**Brush has no appearance model.** Checked `brush-cli --help`: there is no
appearance embedding, no per-image exposure or white-balance latent. So if a
differently-lit flight *did* register, the optimiser would have to explain
contradictory colours at the same 3D point using geometry alone - it either
averages them (washed out) or grows view-dependent floaters. That is the same
mechanism that produces the sky blobs, applied to the whole scene.

On top of that this is an **active construction site**: raw graded dirt in May,
apparently landscaped with paths and plantings by August. Changed geometry
cannot be averaged into one static model at all.

### What to do

Fold in **0027 only**: same session, same light, same scene state, strong
overlap, and `colmap/site/database.db` already holds its features and the
47 586 match pairs. It needs a mapper re-run with the hedge BA settings **and**
snapshots (see the killed-mapper section), not the defaults that ran 9 h 38 m
and produced nothing.

### WARNING: --strip-floaters does not transfer to the city scene

The AGL filter removes splats far above local terrain. On a hillside that is
exactly the sky blobs. **In downtown SF the towers ARE the structure high above
terrain** - running it there would delete Salesforce Tower before it deleted
any sky. Do not use `--strip-floaters` on the `sf` reconstruction.

The distance-to-COLMAP-points rule that failed on the site may actually be the
right one downtown, for the same reason it failed there: it is really a texture
measure. Bare dirt yields no features, so real ground looked like a floater.
Glass-and-window facades are densely featured, so real building surface will sit
close to triangulated points while sky blobs will not. Verify before trusting.

`--sky-margin` (drop above highest camera + margin) is still valid downtown, but
the margin must account for the drone flying *below* tower tops - check
`camera height max` against the tallest structure before setting it.

## Site v2: 0026 + 0027 combined

The mapper re-run that the first attempt failed at. Same database (features and
matches were already there), but with the hedge BA settings **and** snapshots.
**23:06 -> 01:55 = 2 h 49 m**, one merged model, no fragmentation.

| | 0026 only | 0026+0027 |
|---|---|---|
| registered | 1 144 (99.0 %) | **2 450 / 2 468 (99.3 %)** |
| 0027 | - | 1 306 / 1 312 (99.5 %) |
| 3D points | 620 600 | **1 276 675** |
| mean reproj | 0.8755 px | 0.9099 px |
| track length | 5.20 | 5.46 |

Training: 30 k iters, cap raised 6 M -> 8 M. **01:57 -> 05:01 = 3 h 04 m**, and
it ended at **7 370 243 splats - under the cap**, so densification stopped on
its own instead of being clipped. Raising it was the right call:

| eval (held-out) | 0026 only | 0026+0027 |
|---|---|---|
| 20 k mean | 22.29 | **22.59** |
| 30 k mean | 22.44 | **22.49** |
| 30 k min | 16.39 | **17.96** |
| 30 k max | 26.30 | **26.48** |

Equal-or-better on every figure while covering 2.1x the scene.

### Brush's `Vertical axis` is unreliable - derive up from geometry

It differed completely between the two scenes:

| scene | Brush axis |
|---|---|
| 0026 only | -0.2459, -0.2482, +0.9370 |
| 0026+0027 | +0.1488, -0.6859, +0.7123 |

On the combined scene it put **90 % of cameras below the terrain**, which is
impossible, and smeared splat AGL (p95 5.06 vs 0.14). Three independent
estimators - minor axis of camera positions, mean camera-down axis, and a
RANSAC ground-plane normal - agreed with each other within 4-8 deg on both
scenes and disagreed with Brush.

`scripts/check_up.py` scores a candidate up by the two things that must hold:
cameras above terrain, and splat AGL concentrated at 0 with a thin high tail.

| scene | up | cams below | splat p95 | p99/p95 |
|---|---|---|---|---|
| 0026 only | Brush | 5 % | 0.14 | 61.6 |
| 0026 only | **geometric** | **4 %** | **0.07** | **89.1** |
| combined | Brush | 90 % | 5.06 | 4.6 |
| combined | **geometric** | **1 %** | **0.36** | **21.5** |

`ply2splat.py --up x,y,z` overrides it. Note the vector passed is the **down**
(gravity) direction, matching Brush's own convention: it is mapped to (0,-1,0)
so that up becomes +Y, which is what the AGL filter assumes.

Two traps worth remembering:
- **Rendering from a capture camera cannot detect a wrong up.** Levelling
  rotates the scene and the cameras together, so the image is identical. Both
  candidates rendered pixel-identically. Only the filter and the viewer's sense
  of "up" while flying are affected. Judge it numerically, not visually.
- RANSAC over 1.27 M points OOM-killed the process. Subsample first.

### Files (all rebuilt from the combined model)

| file | splats | size |
|---|---|---|
| `site_path.splat` | 3 054 225 | 93 MB | filtered + flight path |
| `site_clean.splat` | 3 000 000 | 92 MB | filtered, no path |
| `site_web.splat` | 1 500 000 | 46 MB | for embedding |

Path radius dropped 0.012 -> 0.004: with 2 450 poses instead of 1 144 the track
was thick enough to hide the site.

```
open "http://localhost:8777/index.html?url=site_path.splat&cam=8"
```

## Downtown San Francisco

Clips 0023 / 0024 / 0025 (2026-05-15, one session). Clips 0014 / 0015 were
excluded: **their gyro is fake** - `org_quat` is identity for all 14 313 and
4 207 samples. Gyroflow reports success and emits one sample per frame, so
nothing fails loudly; `select_frames2.py` silently fell back to the kmax time
cap and produced plausible-looking counts. There is now a guard that refuses
constant-gyro clips. They are also a different sensor mode (h264 2688x2016 vs
hevc 3840x2880) needing their own camera model.

### COLMAP: 100 % registered

2 825 frames, 8 deg threshold with a 200 ms max gap (denser than the site,
as insurance against repetitive facades).

| | site (0026+0027) | SF downtown |
|---|---|---|
| registered | 99.3 % | **100.0 %** |
| 3D points | 1 276 675 | 994 472 |
| mean reproj | 0.9099 px | 0.9679 px |
| **track length** | 5.46 | **10.43** |

The worry about glass and repetitive windows was wrong. Downtown is *easier*
than a dirt hillside: median 12 190 SIFT features per image and a mean track
length nearly double the site's, because building features are visible from far
more viewpoints. Camera self-calibrated to fx 725.17 / k1 -0.10637 against the
site's 725.61 / -0.10878 - independent confirmation of the shared camera mode.

Features 12.4 min, sequential 26.1, cross-clip 20.3, mapper 3 h 16 m.

### Training and what actually limits it

30 k iters, 10 M cap, **3 h 01 m**, finished at 6 634 712 splats (under cap).

| iter | mean | median | max |
|---|---|---|---|
| 5 000 | 16.83 | 15.71 | 24.25 |
| 15 000 | 17.45 | 16.36 | 28.67 |
| 30 000 | **17.89** | 16.98 | **29.22** |

Far below the site's 22.49 - but the mean is misleading. Splitting the eval set:

| iter | frames <20 dB | frames >=20 dB |
|---|---|---|
| 5 000 | n=61, 15.90 | n=10, 22.53 |
| 15 000 | n=59, 16.13 | n=12, 23.93 |

Ground-level shots in Salesforce Park hit **26.5 dB and are near
indistinguishable from ground truth** - individual blades of grass. The low
group is not primarily the glass: looking at median frames, the tower shape and
street grid are correct and **large semi-transparent haze blobs float in front
of the scene**. Downtown has far more sky in frame and a much larger depth
range (point spread p50 5.1, p99 44.7) for unconstrained Gaussians to hide in.

The genuinely unfixable part is the mirrored tower surface: a glass cylinder
reflecting the whole city has no view-independent appearance for splats to
represent. SH models soft view-dependence, not reflections.

### --strip-far: the mirror image of --strip-floaters

The distance-to-COLMAP-points rule that **deleted 14.6 % real terrain on bare
dirt is the correct rule downtown**, for exactly the reason it failed there: it
measures texture, not floatiness. Bare dirt has no features so real ground looks
far; glass-and-window facades are dense with them.

| | near points (d<p50) | far (d>p90) |
|---|---|---|
| median size | - | **0.324** |
| blueness | +0.012 | **+0.111** |
| rgb | 0.50 0.53 0.52 | 0.32 0.40 0.48 |

Distance is tightly concentrated (p50 **0.027**, p75 0.077) then a long tail
(p95 2.46, p99 9.09). `--strip-far` drops 9.46 % and the haze goes.

**Never use `--strip-floaters` (AGL) downtown** - the towers are the structure
high above terrain and it would delete them first.

### Up axis: Brush writes it with Z negated

Third scene, same relationship - Brush's `Vertical axis` matches the geometric
up in X and Y and is **negated in Z**:

| scene | Brush axis | camera-PCA minor |
|---|---|---|
| 0026 | -0.2459, -0.2482, **+0.9370** | -0.246, -0.248, **-0.937** |
| 0026+0027 | 0.1488, -0.6859, **+0.7123** | 0.149, -0.686, **-0.712** |
| SF | 0.3160, 0.9420, **-0.1128** | 0.316, 0.942, **+0.1128** |

**Caveat, stated plainly:** the up axis was validated rigorously on the site
scenes (`check_up.py`: cameras above terrain, splat AGL concentrated). For SF
that test does not apply - the lowest quartile of a city is building bases and
far terrain, not a street plane (flatness 0.37, so the fit is meaningless).
The camera-PCA axis is used because it matches the pattern above and agrees
with a ground-plane fit to 12 deg, but a residual tilt of up to ~12 deg while
flying has NOT been ruled out. It does not affect renders from capture cameras,
which are invariant to levelling.

### Files

| file | splats | size |
|---|---|---|
| `sf.splat` | 3 000 000 | 92 MB | haze filtered |
| `sf_path.splat` | 3 042 973 | 93 MB | + flight path |
| `sf_web.splat` | 1 500 000 | 46 MB | for embedding |
| `sf_cameras.json` | 63 poses | | |

```
open "http://localhost:8777/index.html?url=sf.splat&cameras=sf_cameras.json&cam=38"
```

## Ghost replay

Fly the flights back through the reconstruction they built, two or three at
once, on a shared clock. Chase cam, onboard cam, comet trails, and a readout of
how far apart in *time* the flights were at the same place.

The point is that every consumer scanning pipeline (Polycam, Luma, KIRI,
Postshot) treats video as an unordered bag of images and throws the flight
away. We already have the poses for every frame, so the flight is recoverable
for free -- and it is the part nobody else has.

```
open "http://localhost:8777/index.html?url=site_clean.splat&cam=8"
open "http://localhost:8777/index.html?url=sf_web.splat&cameras=sf_cameras.json&traj=sf_traj.json&cam=38"
```

Use `site_clean.splat`, not `site_path.splat` -- the latter has the flight path
baked in as Gaussians and you would see it twice.

| key | |
|---|---|
| `space` | play / pause |
| `C` | camera: free -> chase -> onboard |
| `N` | which flight the camera follows |
| `[` `]` | scrub 5 s |
| `,` `.` | speed 0.125x .. 8x |
| `R` | restart |
| `T` | trails |
| `G` | hide ghosts entirely |

`?traj=`, `?mode=` (0/1/2) and `?dronescale=` override the defaults;
`window.__ghosts` is a live console handle.

### scripts/trajectory.py

Emits `<scene>_traj.json`: position, orientation and speed resampled onto a
uniform 30 Hz clock, in the viewer's frame.

**Do not re-derive the viewer transform -- recover it.** `ply2splat` levels by
`Rlev`, centres on the camera centroid and scales by `sfac`, and reproducing
that means knowing exactly which `--up` was passed on the day. `cameras.json`
is already in the viewer frame, so both fall straight out of it:

    Rn = Rlev @ R_c2w         ->  Rlev = Rn @ R_c2w.T    (exact, from one camera)
    Cn = (C @ Rlev.T - c)*s   ->  least squares for s and -s*c

Residual is 3.6e-15 on the site and 5.3e-15 on SF, i.e. machine precision. The
script refuses to run if it exceeds 1e-3, which also catches being pointed at
the wrong sparse model.

**Time comes from the frame index, not the pose index.** Selection was adaptive
(constant inter-frame rotation), so poses are dense in the turns and sparse on
the straights -- 5.8 to 9.1 Hz mean across the five clips. Play them back at a
fixed rate per pose and the drone crawls through corners and teleports down
straights. `0026_004800.jpg` is frame 4800 at 50 fps, so t = 96.0 s. That one
detail is the difference between an animation and a replay.

### The camera mount angle falls out of the flight

An FPV camera is bolted on tilted up, so camera forward is not body forward:
render the drone in the camera frame and it sits permanently nose-up by the
mount angle. It can be estimated as the median angle, in the camera's vertical
plane, between camera forward and the direction of travel.

Two filters are load-bearing. Only samples above the median speed count
(direction of travel is meaningless at a hover), and only *level* ones -- climb
vertically while looking out at a building and the angle reads ~90 deg and
poisons the median.

| clip | est. tilt | IQR | |
|---|---|---|---|
| 0026 | **+20.0 deg** | 11 | site |
| 0027 | **+19.8 deg** | 15 | site |
| 0023 | +0.0 | 43 | rejected |
| 0024 | +0.0 | 47 | rejected |
| 0025 | +0.0 | 28 | rejected |

Two independent flights agreeing to **0.2 deg** on a quantity never measured is
a real cross-validation. The downtown clips are slow and full of verticals and
do not constrain it at all, which is why `--tilt-iqr` (default 25 deg) rejects
them rather than believing 49/26/16. SF is built with `--tilt 20` borrowed from
the site; same aircraft, sequential clip numbers, but it is *borrowed, not
measured*.

### Occlusion: why the obvious approach fails

The viewer sorts and alpha-blends with no depth buffer -- `gl_Position.z` was a
hardcoded `0.0`. That is why the flight path had to be baked into the `.splat`
as Gaussians in the first place. Moving geometry cannot do that: the sort is a
throttled pass over millions of splats, not something to redo per frame.

The obvious fix -- draw the drone, then splats over it -- destroys the image.
The splat pass blends `ONE_MINUS_DST_ALPHA / ONE`: front-to-back *under*
compositing, where dst alpha is accumulated coverage. Anything drawn beforehand
with alpha 1 makes every subsequent splat multiply by `(1 - 1)` and the entire
scene goes black.

What works is drawing the ghosts twice, on either side of the splat pass:

    clear colour+depth
    -> ghosts DEPTH ONLY, colour masked off       stakes out the depth buffer
    -> splats, blended, depth TEST on / WRITE off splats behind a ghost are
                                                  rejected and never accumulate
    -> ghosts in colour, same "under" blend        composited beneath exactly
                                                   the splats in front of them

The second pass gets correct occlusion *for free from the blend operator* -- a
ghost is dimmed by precisely the coverage accumulated in front of it -- and the
prepass is what guarantees nothing behind it ever contributed. `main.js` now
writes real (clamped) NDC depth from the splat pass; splats still never write
depth, so splat-vs-splat ordering is untouched.

Verified both directions: disabling the prepass makes the ghosts vanish
completely (coverage saturates and multiplies them to zero), and pushing a
drone 0.55 below its real path makes it disappear under the terrain.

### Trails: two traps at the near plane

The trail is the whole trajectory uploaded once as a ribbon, so animating the
growing tail is just moving the `drawArrays` range -- no per-frame buffer churn.

Widening it in **screen space** (the standard constant-pixel-width trick) needs
a divide by clip `w`, and in chase view the tail runs right past the camera
where `w <= 0`. The offsets blew up into viewport-sized triangles which, in the
depth prepass, staked out the near plane across the whole viewport and rejected
every splat behind them. **The entire scene went black.** Widening in view
space instead has no divide to explode and is clipped correctly.

Even then, a constant *world* width is ~7 % of the viewport where it crosses
`znear` (0.2), so onboard -- where the head of the trail sits exactly at the
camera -- it flared into a wedge across the view. Fixed by tapering the width
to nothing over the last 0.55 units and fading alpha with view distance.

### What the flights actually share

| | within 0.15 | within 0.3 | time offset there |
|---|---|---|---|
| site 0026 vs 0027 | **58.3 %** | 72.9 % | median -0.2 s, IQR 66.9 s |
| SF 0023 vs 0025 | 44.7 % | 76.8 % | median -51.8 s, IQR 43.1 s |
| SF 0024 vs 0025 | 54.7 % | 81.6 % | median +89.4 s, IQR 144.0 s |

The two site flights genuinely re-fly most of the same route, so the ghost
comparison is real. But they share no start gate -- the IQR of the time offset
is 67 s -- so there is no global "who is winning". The HUD therefore reports
the honest local quantity: at the active drone's position, how far apart in
time the other flight was *at that place*, and if it never came within 0.35
units, the distance instead.


## Aug-27 session: why it could not extend the May model

Eleven new clips of the same hilltop, shot 2026-08-27 16:35-17:15 (the original
0026/0027 are 2026-05-17 11:38). The ask was to fold them into the existing
render. **They cannot be, and lighting is only the second reason.**

### The overlap test, and the trap in it

| test | 0026 vs 0027 (same session) | Aug vs May, best of 5 clips |
|---|---|---|
| peak SIFT inliers | **302** | 50-76 |
| pairs >= 80 inliers | 7 | **0, in any clip** |
| within-session control | median 13, max 57 | - |

The first pass, at 18 probes per clip, reported a flat "no usable overlap"
(max 29-43). That was **under-sampled**: re-probing at 40 frames per new clip
against 60 spread across the old session found a peak of 76. A sparse negative
is not a negative. Probe densely before concluding.

But 76 against a control ceiling of 57 is not a real link either, and the
reason is visible only if you ask *where* the inliers are:

```
OLD frame  76 of 77 inliers inside the horizon band   (y 916-1313 of 1440)
NEW frame  97 % of inliers above its horizon line     (y median 435 of 1440)
```

**Every inlier is on the distant suburb skyline. None are on the ground the
drone is flying over.** Far-field features constrain orientation and give
essentially no triangulation baseline, so they cannot join two models. Checking
the y-distribution of the inliers against the horizon costs one script and
settles what an inlier count alone cannot.

### The site itself changed

`reports/xdense/0050_top1_76.jpg` is the side-by-side. May: rocky, uneven
ground under hard sun. August: smooth graded dirt, fresh tracks, a bulldozer
parked in frame. Same hilltop, different shape - an active earthworks site.
No single static model can average two different geometries, so even a good
feature link would not have helped. Brush having no appearance embedding (hard
sun vs flat overcast) was the *third* problem, not the first.

### Two format inferences that were wrong

- The guard comment in `select_frames2.py` blamed the fake-gyro trap on
  "h264 2688x2016". These new clips are exactly that format and their gyro is
  perfect (14 265 unique quaternions in 14 265 samples). The old failure was
  clips 0014/0015 specifically, not the mode.
- `run_sf.sh` said 2688x2016 was "a DIFFERENT sensor mode needing its own
  camera model". The deciding field is the FOV tag, not the pixel count:
  0014 is 2688x2016 **FOV:MAX**, while 0050-0060 are 2688x2016 **FOV:Normal**,
  the same field of view as the 3840x2880 clips. Read it with
  `ffprobe -show_entries format_tags=comment`.

Both comments are corrected. The lesson is the same one twice: test the
property, do not infer it from the recording format.

### The new session is better raw material

| | May (0026+0027) | Aug (9 clips) |
|---|---|---|
| selected frames | 2 468 | 7 471 |
| sharpness (Laplacian var @1920x1440) | 301 | ~1 100 |
| light | hard sun, blue sky | flat overcast, 40 min span |
| codec | HEVC 120 Mbps / 11.1 MP | h264 100 Mbps / 5.4 MP |

Roughly double the bitrate per pixel is the likely reason the lower-resolution
footage measures sharper at working resolution. Laplacian variance is
content-sensitive, so treat that as indicative rather than decisive.

So the August footage is built as its **own** scene, not an extension.

### The Aug session is one connected area (36/36 pairs)

All-pairs matchability over the 9 usable clips, 18 probes each
(`reports/matrix_aug.txt`), grouped by hilltop (0050/51/52/53/55) vs cemetery
(0057/58/59/60):

| group | pairs | best | weakest | pairs >=80 | weakest link |
|---|---|---|---|---|---|
| within-hilltop | 10 | 413 | 59 | 66 | 0051-0055 |
| within-cemetery | 6 | 337 | 94 | 16 | 0057-0058 |
| **hilltop <-> cemetery** | 20 | 236 | **84** | 56 | 0050-0058 |

Every measured pair reaches max >= 40, so the whole area is a single connected
graph. The number that matters: **the weakest hilltop-cemetery link (84) is
stronger than the best Aug-vs-May link (76)**. The same test that says these
two halves belong in one model says the two sessions do not.

### Plan: two scenes, then a merged one

`frames/aug` 3 777 frames (hilltop), `frames/aug2` 3 694 (cemetery).
`scripts/run_aug.sh` and `scripts/run_session.sh aug2 aug2 "0057 0058 0059 0060"`.

To combine, do NOT build both and then `colmap model_merger` -- that needs the
two models to share registered images, and disjoint frame sets never do.
Instead:

```
colmap database_merger --database_path1 colmap/aug/database.db \
  --database_path2 colmap/aug2/database.db \
  --merged_database_path colmap/augall/database.db
# then import hilltop x cemetery pairs, then ONE mapper run over the merged set
```

Feature extraction is the expensive per-image step and it is reused; the merge
costs a mapper run, not a restart. Note the merged model is solved in its own
frame, so `cameras.json`, the `.splat` export and `trajectory.py` all have to
be regenerated for it -- trajectory.py recovers the transform from cameras.json
rather than assuming one, so it needs no code change, just a re-run.

### Undistorting the footage overlay (2026-08-27)

The PiP was the raw fisheye while the render is a pinhole projection, so the
two only agreed near the optical axis. Fixed by undistorting the proxies with
the COLMAP OPENCV coefficients into **the same pinhole camera the render uses**
(`newCameraMatrix = K`, not `getOptimalNewCameraMatrix`) -- any other target
camera would leave the footage misaligned, just differently.

`scripts/make_proxy.sh <clip> <src> fx fy cx cy k1 k2 p1 p2` builds one, driving
`scripts/undistort_filter.py` between two ffmpeg processes (VideoToolbox decode
-> scale 640x480 -> cv2.remap INTER_CUBIC -> libx264 crf 30). ~70 s per 145 s
clip. The raw proxy is kept as `<clip>_raw.mp4`; **U** toggles between them in
the viewer.

Measured on frame `0026_002850` (9,067 features with triangulated 3D points),
distance from where the render projects the 3D point to where the feature
actually sits in the footage, in 640-wide proxy pixels:

| region | raw | undistorted |
|---|---|---|
| all | median 15.9, p90 63.8, max 171.7 | median 0.29, p90 0.86, max 8.6 |
| inner third | 1.5 | 0.23 |
| middle third | 18.3 | 0.25 |
| outer third | 72.3 | 0.89 |

Verified end-to-end against the *live* viewer too (`scripts/aligncheck.py` writes
`viewer/aligncheck.json`; the page projects those 3D points through its own
view+projection matrices): median 0.57 px, p90 1.7 px at 960x720 over 40 probes
spread to the frame corners.

Traps hit while checking this:

- **ffmpeg's `lenscorrection` is not the OpenCV model.** It normalises radius by
  the half-diagonal rather than the focal length, so coefficients need rescaling
  by `(R/f)^2` and `(R/f)^4`, and it has no tangential term at all. The remap
  route applies the actual model.
- **k1 < 0 means the undistorted clip is slightly narrower than the raw one**
  (~12% off each side here), not letterboxed. Every output pixel samples from
  inside the source; the outer sliver is simply outside the pinhole frustum the
  render draws.
- COLMAP puts the principal point at `W/2`, OpenCV at `(W-1)/2`. Half a pixel at
  1920, but the conversion belongs in the script.
- **A screenshot taken right after a seek shows the previous frame.** Two
  separate "the overlay doesn't match!" panics were stale PiP frames; a third
  was eyeballing a 800x600 downscale where the PiP covers the very region it
  should be compared against. The numeric probe is the only honest check.
- `colmap/aug/sparse/0` is a **4-image junk model** the mapper wrote out and
  abandoned (its camera converged to fx 606 / fy 294). The real August model
  will land in a later-numbered directory -- do not grab `sparse/0` by habit.

**August proxies** (2026-08-28): built from `colmap/aug/sparse/1`'s converged
intrinsics `721.46372 718.83606 960 720 -0.11048 0.00859 0.0007 0.00027` -- not
May's, and emphatically not `sparse/0`'s. Clips 0050/0051/0052/0053/0055, plus
`_raw` counterparts for the **U** toggle; 223 MB for the whole `viewer/video/`.
Frame count and duration verified equal to the source on every one, since a
drift there would desync the overlay silently rather than visibly.

## August hilltop scene (2026-08-28)

`colmap/aug/sparse/1` (3777/3777, 1.60 M points, 0.80 px) -> Brush 30 k iters,
4 h 30 m, 8 M splats (hit the `--max-splats` ceiling between iteration 10 k and
20 k, which is why the 20 k and 30 k PLYs are the same size).

| file | splats | size |
|---|---|---|
| `viewer/aug_clean.splat` | 3 000 000 | 92 MB |
| `viewer/aug_web.splat` | 1 500 000 | 46 MB |
| `viewer/aug_cameras.json` | 3777 poses | |
| `viewer/aug_traj.json` | 5 clips, 27 217 samples | 2.9 MB |

```
open "http://localhost:8777/index.html?url=aug_web.splat&cameras=aug_cameras.json&traj=aug_traj.json&cam=200&mode=2"
```

### The up axis had to be re-derived, not copied

Brush's `Vertical axis` comment was 40 deg off (its Z is negated -- **fourth**
scene in a row). Three independent methods agreed on the correction:

| method | result |
|---|---|
| mean camera-down axis | 1.9 deg from the Z-flipped vector's negation |
| camera-position minor axis | 0.0 deg from the Z-flipped vector |
| cameras vs point cloud | cameras sit on the +zflip side |
| `check_up.py` | 0% cameras below terrain, splat AGL p95 0.07, separation 59 |

**The sign is opposite to May's**, because a PCA eigenvector's sign is arbitrary,
so the May `--up` could not be copied across. Floater filter then dropped 2.22%.

### Mount tilt: trust the session, not the clip

Per-clip estimates were +17.6 / **-24.9** / +18.5 / +9.9 / +18.8. The -24.9 is
physically impossible (this camera tilts *up*) and came from a 9-second clip
with 35 poses; its IQR was exactly 25.0, so the default `--tilt-iqr 25` did
**not** reject it (the test is `>`, not `>=`). The 38 s clip gave +9.9.

The three long clips agree at 17.6 / 18.5 / 18.8 over 10 569 level samples, and
one mount on one day cannot change between clips, so the session-consensus
`--tilt 18.3` (sample-weighted) is used for all five. Rule: **estimate tilt per
session, not per clip, and treat any clip under ~60 s as unable to constrain it.**

### cameras.json was clobbered, and how it was rebuilt

`ply2splat --cameras-out` defaults to `cameras.json` **next to the output
splat**. A second run written into `viewer/` without that flag therefore
overwrote the May scene's `viewer/cameras.json` with 3777 August cameras.
Always pass `--cameras-out` explicitly for a named scene.

It was rebuilt without guessing the original flags. The viewer transform has
exactly one unknown, `Rlev`; `c` and `sfac` are then derived from the COLMAP
cameras. `site_traj.json` stores `Cn` on a resampled grid, and wherever a source
frame time `idx/fps` coincides with a grid time `t0 + k*dt` the stored sample IS
the transformed camera centre (`np.interp` returns the node value at a node) --
499 exact correspondences. Fitting a similarity transform to those recovered
scale 0.458316 and offset [-0.1122 -0.0085 0.1101]; a second estimate of `Rlev`
from the stored `cam_quat` agreed to **0.000136 deg**. `scripts/rebuild_cameras.py`.

The stored positions are rounded to 4 decimals, so ~5e-5 is the quantisation
floor of that fit, not an error -- do not set the accept threshold below it.
Verified end-to-end by re-running `aligncheck.py` against the live viewer:
median 0.57 px, p90 1.70 px, identical to the pre-clobber numbers.

**That check must be run at 4:3.** At 1280x720 it reported a median of 108 px,
which is purely the viewer scaling focal off viewport height (vertical FOV held,
horizontal widened) -- not a transform error.

## The ground is a slab, not a surface

Reported symptom: flying low, the bottom of the frame sees *through* the
ground, and the ground "extends under what the actual surface is". Both halves
of that are literally true and measurable.

`scripts/raster.py` reproduces `viewer/main.js`'s shader offline (the Claude
browser pane cannot render the WebGL viewer, so every check here is numerical).
`scripts/ground_thick.py` measures the vertical opacity profile over open
ground columns:

| | cemetery (aug2) | hilltop (aug) |
|---|---|---|
| 10%->90% opacity transition | 0.270 = **144%** of median flight AGL | 0.360 = **131%** |
| opacity mass **below** the visible surface | **49.1%** | **47.0%** |
| ground splat opacity, median | 0.267 (83% under 0.5) | 0.255 (83% under 0.5) |
| thin axis vs surface normal, median | **46.8 deg** | **48.5 deg** |
| thin axis within 20 deg / beyond 60 deg | 14.4% / 33.8% | 13.6% / 36.1% |

Half the ground's substance sits *under* the surface, and only one splat in
seven has its flat face parallel to the ground.

**Why it only shows up close.** Accumulated alpha is `1 - prod(1-a_i)` along the
ray, so it depends on PATH LENGTH through the slab. The vertical optical depth
`tau = sum -ln(1-a_i)` is only about 1, so a ray looking straight DOWN exits
~60% opaque, while a grazing ray travels far enough inside to saturate. Hence
distant ground looks solid and the near ground at the bottom of the frame does
not. Measured bottom-third accumulated opacity: **0.996 at mid altitude, 0.634
when low**.

**Why 3DGS produces this.** There is no surface prior. From grazing FPV views a
diffuse semi-transparent slab and a crisp opaque surface have identical loss.
Chase footage is the worst case for this: the ground is seen almost only at
grazing incidence from one direction of travel, which is exactly the geometry
that fails to constrain position along the ray.

### Two plausible causes that measure as innocent

**The viewer's hardcoded `znear = 0.2`** is 83% of median flight AGL on the
cemetery scene, and `vColor *= clamp(ndc_z + 1, 0, 1)` fades a splat to *zero*
by z=0.100. Ray-casting the terrain says 82% of bottom-third ground hits are
inside that band. It still changes nothing: the fade keys off each splat's
CENTRE depth, and those pixels are covered by splats centred further away.
Lowering znear to 0.01 moved bottom-third opacity by <0.03.

**Brush trains with a screen-space low-pass the viewer omits.**
`crates/brush-render/src/kernels/helpers.rs::compensate_cov2d` adds
`cov_blur = 0.3` to the cov2d diagonal (0.1 under mip-splatting) in the
forward, visible AND backward kernels. Both PLYs record
`comment SplatRenderMode: default`, so 0.3 with no opacity compensation.
`main.js` adds nothing -- a genuine trainer/viewer mismatch. But ground splats
here are several pixels across, not sub-pixel, so applying it moves
bottom-third opacity by <0.001. Fix it for correctness; it is not the cause.

### Flattening alone makes it WORSE

The obvious fix -- collapse the slab into a thin shell -- regressed the metric:
bottom-third opacity 0.797 -> 0.513, holes 18.9% -> 49.8%. Thinning shortens
every path through the ground. **The mass was never the problem, the optical
depth is.** Raising it is a per-splat operation with an exact meaning:

    a -> 1 - (1-a)^k        multiplies tau by exactly k, for every ray direction

so it cannot be defeated by viewing angle, unlike anything geometric.

### Rasterizer gotcha (cost an hour)

`|majorAxis| = min(sqrt(2*lambda),1024)` and a quad vertex at `position.x = +-2`
lands `position.x * |majorAxis| / 2` **pixels** out, so `exp(-|position|^2)` is a
Gaussian of pixel std **`sqrt(lambda)/2`**, not `sqrt(2*lambda)/2`. Getting that
wrong renders every splat sqrt(2) too wide and overstates scene opacity --
it hid the problem (0.793 vs the true 0.634).

Also: the web/NAS build caps 3M -> 1.5M by `opacity x projected area`, which
keeps only **44.8% of ground splats against 56.5% of everything else** because
ground splats are faint. The hosted scene's ground is thinner than the local one.

### What actually fixes it (and what provably does not)

Measured on `aug2_web.splat` at 960x720, bottom third of level low-flight
frames, `scripts/measure.py` / `scripts/target_test.py`:

| intervention | LOW opacity | LOW holes | MID opacity |
|---|---|---|---|
| original | 0.634 | 35.9% | 0.975 |
| ground opacity k=2 | 0.682 | 30.1% | 0.990 |
| ground opacity k=3 | 0.706 | 25.3% | 0.995 |
| ground opacity k=4 | 0.721 | 23.2% | 0.996 |
| **ALL splats k=3 (ceiling)** | **0.808** | **13.1%** | 0.996 |
| flatten v1 (thin 0.5, weak boost) | 0.483 | 53.5% | 0.988 |
| flatten v2 (align + grow 1.25 + k=4) | **0.421** | **61.3%** | 0.997 |

**Every geometric intervention made it worse.** The reason is not obvious and
is worth writing down: **the slab's thickness is what currently provides
surface coverage.** A downward ray crosses a median of 24 splats, each
contributing only ~0.018 because it clips them in the Gaussian tail. Collapse
those 24 onto one plane and the ray now crosses 2-5 -- and wherever the
in-plane tiling has a gap, nothing covers the pixel at all. Thickness is
masking gaps. It is a bad representation that is load-bearing.

Diagnostic that settles it (`scripts/hole_anatomy.py`): only **2.4%** of the
bottom third has fewer than 3 splats on the ray, while **33.8%** reads as a
hole. So this is NOT missing geometry, and it is NOT fixable by rearranging
what is there.

Opacity is the only post-hoc lever that helps, and it plateaus at 0.808 even
when applied to every splat in the scene (an unusable config -- it makes the
sky opaque). **Conclusion: this cannot be fixed post-hoc. It belongs in
training.**

Brush has **no depth or normal regulariser, no 2DGS mode, no flatness prior**
(`crates/brush-train/src/config.rs` is the complete list of knobs). The
relevant ones are `--opac-decay` (0.004), which pushes splats toward opaque-or-
pruned; `--render-mode mip`, which switches `cov_blur` to 0.1 AND applies the
opacity compensation `filter_comp = sqrt(det_raw/det_blurred)`; and
`--growth-grad-threshold` (0.0025), which currently starves the textureless
ground of densification because growth follows image gradient.

The band matters more than the exponent. Widening the ground class to
`h in [-0.6, +0.15]` (76.8% of splats, no smoothness/openness test) with k=4:

| camera | before | after |
|---|---|---|
| aug2 cam 10 | 0.683, 28.5% holes | 0.760, 23.2% holes |
| aug2 cam 2999 | 0.526, **54.7%** holes | 0.738, **18.9%** holes |

Visibly better on the worse frames (`work/opacity_ab.png`) and free -- no
retraining, no geometry edit, reversible. It is a mitigation, not a fix: the
ceiling is ~0.81 and MID-altitude views were already 0.975+, so all it buys is
the low-altitude case.

### aug2 up axis, settled

The cemetery scene's `--up` for `ply2splat` is **`-0.104898565,0.56495816,0.8184245`**
-- the FULL NEGATION of the Brush `Vertical axis` comment, not the comment and
not the comment with Z negated. Using the Z-negated form put **94.2% of cameras
below the terrain** (baseline: 0.0%).

Verify with the two-line check, never by eye: rebuild `--cameras-out` and diff
the positions against `viewer/aug2_cameras.json`. Correct up reproduces the
baseline frame to ~6e-6, because `ply2splat` centres and scales on the CAMERA
positions and those are identical across runs of the same COLMAP model. A wrong
sign shows up as a max diff of ~8. `scripts/eval_run.sh` does this and asserts.

### The flatness prior cannot fix the slab (measured, not argued)

`--flatten-strength` works exactly as designed -- verified with a deliberately
over-strength A/B at 1000 iters (`work/knob_test.sh`, `scripts/ply_scales.py`):

| strength | flatness | s_min | s_max |
|---|---|---|---|
| 0.0 | 0.7878 | 0.00475 | **0.00625** |
| 0.3 | 0.7411 | 0.00447 | **0.00625** |

`s_max` identical to 5 dp while `s_min` and flatness move by the same factor:
the largest axis is untouched, as specified.

**But it addresses the wrong quantity.** On the cemetery scene a ground splat's
thin axis is 0.0019 = **1.0% of flight altitude**, while the 10->90% opacity
transition is 0.245 = **125%**. The slab is **129x thicker than one splat**, so
the depth over which opacity accumulates is set by the SPREAD OF SPLAT CENTRES,
not by each splat's thickness. `--flatten-strength` only ever rescales axes --
it never moves a centre. Thinning every ground splat to 39% of its size leaves
the slab essentially unchanged.

The only route by which 3a could help is indirect: a thinner splat that is
misaligned with the surface renders visibly wrong, so the reconstruction loss
gains pressure to rotate it flat. That is how 2DGS gets its normals. Whether
that pressure is strong enough at these strengths is the one open question, and
the running 30k job answers it for free.

**The depth-distortion loss (3b) is the actual fix**, because
`sum_{i,j} w_i w_j |t_i - t_j|` penalises exactly the spread of contribution
ALONG the ray -- the quantity that is 129x too large. Priority accordingly.

### 3a VERDICT: the flatness prior is a documented negative

Stage-matched (30k vs 30k), cap-matched (8M vs 8M), and both PLYs pushed
through the SAME `scripts/eval_run.sh` -- the earlier `aug2_clean.splat`
numbers were rebuilt because their ply2splat flags were no longer recorded.

| | baseline @30k | flatness prior @30k |
|---|---|---|
| flatness s_min/s_max | 0.309 | **0.258** (-17%) |
| thin axis vs normal | 46.8 deg | 47.3 deg (**null**) |
| within 20 deg | 14.3% | 14.0% (**null**) |
| slab 10->90 thickness | 142% | 126% |
| mass below surface | 48.3% | 46.2% |
| held-out PSNR @30k | - | 22.14 dB (healthy) |
| **LOW bottom-third opacity** | **0.654** | **0.617** |
| **LOW holes (<0.5)** | **33.5%** | **38.9%** |

**It makes the reported symptom WORSE.** Holes 33.5% -> 38.9%. Same mechanism
that sank the post-hoc attempt: a flatter splat covers fewer pixels, and the
orientation never improved to compensate, so rays cross less material. The
training-time version is *less* harmful than post-hoc editing (38.9% vs 61.3%)
because densification refills some of the gap -- which was the one thing the
training-time framing genuinely bought.

Note the optimiser partly routed around the constraint: `s_min` fell only 7%
while `s_max` GREW 6%. Flatness improved mostly by inflating the axis the prior
deliberately never touches.

The knob itself is correct and stays in the tree, defaulting to 0.0
(`--flatten-strength`, verified by an over-strength A/B where `s_max` was
identical to 5 dp). It is simply aimed at splat SHAPE, and the defect is splat
ARRANGEMENT. Do not reach for it again on this problem.

### 3b -- depth-distortion loss: implemented and gradient-verified

The 2DGS depth-distortion term is the only candidate that targets the quantity
3a proved was wrong. It penalises the SPREAD of a ray's contribution along its
own depth:

    L_d = sum_{i,j} w_i w_j |t_i - t_j|        w_i = alpha_i * T_i

3a shaped individual splats. This shapes their ARRANGEMENT along the ray, which
is what the 129x scale gap (slab 125% of flight altitude vs splat thin axis
1.0%) says is actually defective.

**Status: correct and complete, not yet tuned.** Forward, backward, loss term
and `--distortion-weight` flag all landed and verified. No training run yet.

#### The O(N) trick

Front-to-back with `t` ascending, `|t_i - t_j| = t_i - t_j` for `j < i`, so the
double sum collapses to two running scalars per pixel:

    A_i = sum_{j<i} w_j     B_i = sum_{j<i} w_j t_j
    L_d = sum_i w_i * (t_i * A_i - B_i)

No extra pass and no sorting -- the rasteriser is already depth-ordered. The
backward gets the prefix sums for free because it replays FORWARD order via
diagonal scheduling, and recovers the suffix sums from per-pixel totals the
forward also emits (channels 5/6), so it needs no second pass either.

#### The part that was not obvious

Alpha reaches the loss by two routes, because `w_i = alpha_i T_i` and every
`T_i` with `i > k` carries a `(1 - alpha_k)` factor:

    route (a), through w_k:    g_k * T_k
    route (b), through T_i>k:  -(1/(1-alpha_k)) * sum_{i>k} w_i g_i

The route-(b) suffix sum is NOT recoverable from the totals, because `g_i`
itself depends on `A_i, B_i`. But it needs no second pass either: it is
maintained the way the kernel already maintains "remaining colour", walking
down from a known total. That total is free --

    R_0 = sum_i w_i g_i = 2L      (Euler: L is homogeneous of degree 2 in w)

and `L` is the per-pixel distortion the forward already wrote to channel 4.

#### Traps, all of which would have failed silently

* **`pix_base` was `pix_id * 4`.** With `out_img` at 7 channels every gradient
  would read the wrong pixel -- with plausible-looking magnitudes.
* **`v_depth_in` had to join the `any_grad` early-out** in
  `project_backwards.rs`. Without it, a splat whose ONLY gradient is depth
  returns before writing anything.
* **`v_combined` went 10 -> 11 lanes unconditionally**, deliberately. A stride
  that varies between two kernels is precisely the silent-corruption mode
  above; the cost is one f32 per visible splat. The atomic WRITE stays
  comptime-gated, so the default path pays nothing, and the read is a no-op
  because the buffer is zero-initialised.
* **The test loss must slice channel 4 ALONE.** Channels 5/6 are saved forward
  state with no gradient path; a plain `img.mean()` hands them upstream
  gradient the backward deliberately ignores, and finite-diff then disagrees
  for a reason that is not a bug.
* Distortion and the C^1 cutoff are orthogonal, so
  `BackwardDistortionSmoothCutoff` had to exist -- `BackwardDistortion` alone
  uses the hard 1/255 step, which central differences straddle.

#### Verification

`tests/distortion_backward.rs`, 13 finite-difference entries, all within
tolerance and most agreeing to 3-4 significant figures:

    means[0][2]     numeric -3.128173e-2   analytic -3.127972e-2
    means[1][2]     numeric  9.220093e-3   analytic  9.219282e-3
    raw_opac[3][0]  numeric  1.171138e-3   analytic  1.171883e-3
    rots[1][2]      numeric -9.918585e-5   analytic -9.876459e-5

`means[*][2]` are load-bearing: mean-z is the only route by which `dL/dt`
reaches a parameter, so they are what proves the new depth path. Two guards
sit beside it -- `distortion_gradient_is_nonzero` (a scene with no overlap
gives L=0 and every gradient 0, which finite-diff would "agree" with
perfectly) and `distortion_does_not_touch_color` (L_d has no colour
dependence, so the SH gradient must be exactly zero).

The default path is unchanged: all 18 pre-existing `finite_diff.rs` tests still
pass, including the depth-extremes and near-camera fuzz cases.

#### Depth normalisation: measured, not needed

`t_k A_k - B_k` is a difference of similar large numbers, so f32 cancellation
scales with depth, and 2DGS normalises depth for exactly this reason. Measured
on the actual scene: radius from centroid p50 **2.56**, p99 **17.4** -- the
reconstruction is unit-normalised, so view depths run ~1-20, the same order as
the test scene's ~3. No normalisation needed. (`L_d` is exactly shift-invariant
in `t`, so per-pixel re-origining stays available if a scene ever needs it.)

This also means 2DGS's published weight range of 100-1000 does NOT transfer:
theirs is on depth normalised to [0,1], ours is world-scale, so the equivalent
weight is smaller by roughly the depth scale. Sweep, do not copy the number.

#### What must gate it

3a is the cautionary tale: PSNR stayed healthy (22.14 dB) the whole way while
the actual symptom got worse. Gate 3b on `scripts/eval_run.sh` plus LOW-altitude
bottom-third opacity and hole fraction -- the numbers that caught 3a -- not on
PSNR.

### Ground opacity boost, shipped into the build

The one intervention that has ever improved this symptom was still
analysis-only (`scripts/opacity_only.py`), so it could not reach an actual
render. It is now `ply2splat.py --ground-opacity-boost K`, off by default.

`a -> 1-(1-a)^K` multiplies optical depth `tau = -ln(1-a)` by exactly K in
EVERY direction, which is why it works on a slab whose problem is path length
rather than geometry. The band is defined relative to the visible surface
(`--ground-band`, default `-0.6 0.15` in viewer units), so trees and sky are
untouched.

Applied AFTER the splat cap, deliberately: boosting first would inflate ground
splats' significance and evict non-ground ones, changing WHICH splats survive.
After the cap the same splats are kept as an unboosted build and only the
ground gets denser, so builds stay comparable.

Measured on the 30k/8M baseline, same cameras, same tooling as the 3a A/B:

| | baseline | K=4 |
|---|---|---|
| LOW bottom-third opacity | 0.654 | **0.689** |
| LOW holes (<0.5) | 33.5% | **29.9%** |
| MID opacity | 0.986 | 0.986 (unchanged) |

Ground band was 1,268,483 splats (42.3%), mean opacity 0.290 -> 0.643.

**Do not confuse this with the earlier 54.7% -> 18.9% figure.** That was one
worst-case frame on `viewer/aug2_web.splat` (1.5M cap, 76.8% ground band); this
is the LOW-cohort aggregate on the 3M eval build. Both are real, they measure
different things, and the aggregate gain is the honest one to quote: holes down
10.7% relative, for free, with no retraining and no cost at altitude.

### Distortion warmup

`--distortion-start-iter` (default 0). 2DGS enables the distortion loss only
after a warmup, and the reason matters: before the geometry has structure,
"concentrate each ray's weight at one depth" is regularising noise and can lock
in bad arrangement. Worth knowing when reading any short sweep of
`--distortion-weight` -- a 1000-iter probe sits ENTIRELY inside the window 2DGS
deliberately skips, so a PSNR hit there does not necessarily predict the 30k
outcome.

#### Choosing K

| K | ground mean opacity | LOW opacity | LOW holes |
|---|---|---|---|
| 1 (off) | 0.290 | 0.654 | 33.5% |
| 2 | 0.458 | 0.672 | 32.2% |
| 4 | 0.643 | 0.689 | 29.9% |
| 6 | 0.742 | 0.698 | 28.6% |
| 10 | 0.843 | 0.707 | 27.7% |

Sharp diminishing returns: K=1->4 buys 3.6 points of hole fraction, K=4->10 only
2.2 more while driving the ground to nearly opaque. That plateau is the same
one the earlier analysis found (ceiling ~0.808 applied to EVERY splat) -- the
boost cannot close the gap on its own, because past a point the remaining holes
are not thin material, they are gaps. MID opacity is 0.986 at every K, so
nothing is over-darkened at altitude.

**K=6 is the default worth shipping**: most of the available gain, ground
opacity still under 0.75.

### `eval_compare.py` was silently scoring nothing

It hardcoded ground-truth paths `frames/site/` and `frames/sf/`, so on ANY
newer frameset it found no ground truth and printed `images : 0` -- which reads
as a result rather than as an error. Every PSNR number this script produced for
an aug2/merged scene before this fix was vacuous. It now searches all of
`frames/*`.

#### Distortion weight sweep (1000 iters, aug2)

| weight | PSNR @1k | vs baseline |
|---|---|---|
| 0 | 16.49 dB | - |
| 0.01 | 16.41 dB | -0.08 (noise) |
| 0.1 | 13.95 dB | **-2.54** |
| 1.0 | 11.04 dB | **-5.45** |

The cliff is between 0.01 and 0.1, confirming that 2DGS's published 100-1000
is off by ~3-4 orders of magnitude for a world-scale scene. At 0.01 the term
barely engages at all (exported PLY size moved 0.24%); at 0.1 it dominates.

Wall-clock cost of the distortion pass: 179s -> 185-195s per 1000 iters, i.e.
**3-9%**. Cheap.

Caveat that shapes how to read this table: 1000 iters is ENTIRELY inside the
warmup window 2DGS deliberately skips, so the damage at 0.1 partly reflects
regularising noise rather than the 30k behaviour. First real run is
`--distortion-weight 0.05 --distortion-start-iter 3000`, chosen near the biting
end on purpose -- 3a showed that a too-weak intervention just buys another
4-hour null result.

### The trainer/viewer `cov_blur` mismatch is dead, measured not argued

Brush's `compensate_cov2d` adds `cov_blur` to the screen-space covariance --
**0.3** in default mode, 0.1 under mip -- and `main.js` omits it entirely. So
splats are trained under a low-pass the viewer never applies, and
`filter_comp` stays exactly 1.0 outside mip mode, so there is no compensating
opacity change either. A real, uncompensated mismatch, and the last remaining
"maybe the viewer is just wrong" explanation. A one-line viewer fix would have
been far cheaper than a kernel, so it was worth killing properly.

Direct measurement of bottom-third splats at low altitude:

| | minor-axis px std |
|---|---|
| p5 / p50 / p95 | 1.82 / **6.29** / 19.92 |

Median minor-axis screen VARIANCE is **39.6**. `cov_blur` adds 0.3 -- under 1%
of it -- and **0.0%** of splats sit below 0.3, i.e. none are in the regime where
the dilate could dominate. Confirmed by rendering the low cohort through
`scripts/dilate_low.py`:

| dilate | LOW opacity | holes |
|---|---|---|
| 0.0 | 0.673 | 30.1% |
| 0.3 (Brush's value) | 0.674 | 30.1% |
| 1.0 (3.3x Brush's value) | 0.675 | 30.0% |

Null at 3.3x the real value. Prediction matched observation. Do not reopen it.

### What the per-pixel arithmetic says the defect actually is

Hole pixels cross ~24 splats, each contributing only ~0.018. With ground
opacity 0.29 that back-solves to `exp(-sigma) = 0.062`, i.e. roughly **2.4
sigma from each splat's centre**. The ray is not starved of geometry -- it
threads the PERIPHERY of many splats instead of the CORE of a few. That is a
statement about where splat centres sit, not about their size or opacity, and
it is the same conclusion the 129x thickness ratio reached, arrived at
independently from the per-pixel side.

It also predicts the opacity plateau exactly: boosting 0.29 -> 0.742 lifts each
tail contribution to ~0.046, so accumulated alpha should reach
`1-(1-0.046)^24 = 0.68`. Measured: **0.698**. The boost behaves exactly as the
model says -- and the same model says it cannot finish the job, which is why
the remaining lever is arrangement.

## MERGE COMPLETE

COLMAP mapper finished after **1,076 minutes** (~18 h).

| | merged |
|---|---|
| registered images | **7,470 / 7,471** |
| models | **1** |
| hilltop / cemetery frames | 3,777 / 3,693 |
| points | 3,359,339 |
| observations | 22,104,422 |
| mean track length | 6.58 |
| mean reprojection error | **0.821 px** |

One image out of 7,471 failed to register, and sub-pixel reprojection holds
across two separate flights in a single coordinate frame. `chain_merged.sh`
fired unattended: validated the model, confirmed both sessions present, waited
for the GPU, and started training at 07:29.

Merged training at 5k, per clip -- the number that proves the merge is real
rather than nominal:

    mean 20.15 dB   (aug2 single-session at 5k: 20.82 dB)
    hilltop  0050 20.12  0051 20.38  0052 19.44  0053 18.42  0055 20.64
    cemetery 0057 21.79  0058 20.05  0059 19.58  0060 19.11

Neither session drags the other down. 0.67 dB below single-session while
covering twice the area with twice the images.

### 3b at w=0.05 -- STRONG NEGATIVE. The loss compressed the scene.

Correct gradients, wrong magnitude. Run: `--distortion-weight 0.05
--distortion-start-iter 3000`, 30k, 8M cap.

| | baseline | 3a flatten | **3b w=0.05** |
|---|---|---|---|
| LOW bottom-third opacity | 0.654 | 0.617 | **0.145** |
| LOW holes | 33.5% | 38.9% | **88.8%** |
| PSNR @30k | - | 22.14 dB | **19.80 dB** |
| PSNR min | - | 14.71 | **5.57** |
| slab thickness | 142% | 126% | 156% |
| scene radius p99 | 17.45 | - | **7.78** |
| ground s_min | 0.00171 | - | 0.00299 (+75%) |

**`dL/dt` did exactly what it was told, globally.** Pulling splats together
along view rays collapsed the reconstruction's depth range -- radius p99 17.45
-> 7.78. The cameras are fixed by COLMAP, so the ground pulled AWAY from the
low-flying cameras and emptied the near field. Splats then grew 75% thicker to
compensate photometrically. Not a gradient bug: finite-diff validated both
signs of the depth gradient before this run.

The re-scored 3a curve reproduced exactly (20.82 / 21.52 / 22.00 / 22.14), so
the comparison is sound.

**What this does NOT establish:** that the loss cannot work. It establishes
that 0.05 is far past the usable range. The 1000-iter probe put the free weight
near 0.01; `scripts/chain_dist_low.sh` retries there, queued behind the merged
training. If 0.01 is null and 0.05 is destructive, the honest conclusion is
that the usable window on a world-scale scene is narrow or absent WITHOUT depth
normalisation -- which is exactly what 2DGS does and what our unit-scale
measurement said we could skip. That measurement may have been the wrong call:
it checked f32 cancellation, not gradient magnitude.

### Splat density does not matter -- a third confirmation of "arrangement"

If the ground were see-through because there were not enough splats, adding
splats would help. Same PLY, three viewer caps, identical cameras:

| viewer cap | LOW opacity | LOW holes | MID opacity |
|---|---|---|---|
| 1.5M | 0.650 | 34.0% | 0.974 |
| 3M | 0.654 | 33.5% | 0.986 |
| 6M | 0.654 | **33.5%** | 0.992 |

**Doubling splat count changes the symptom by nothing.** Halving it costs 0.5
points. This is consistent with the per-pixel finding that hole pixels ALREADY
cross ~24 splats -- more geometry in the same arrangement adds nothing.

Practical consequence: the merged scene covers ~2x the area at the same 8M cap,
so viewer-side density halves. That was worth checking before committing 6 GPU
hours to a higher-cap rerun. It does not need one.

### Per-ray normalisation for the distortion loss

`--distortion-normalize`. `L_d` is degree-1 in `t`, so dividing a pixel's loss
by a DETACHED scale is exactly equivalent to normalising that pixel's `t` -- and
the scale we want, the ray's own mean depth, is already
`B_total / A_total`, both of which the forward saves in channels 5/6. No kernel
change at all.

`detach()` is load-bearing: channels 5/6 carry no gradient path in the backward,
so letting autodiff route through them would build a route the kernel silently
ignores. (The kernel physically never reads `v_output[5..6]`, so it enforces the
stop-gradient regardless -- the detach makes the intent explicit and keeps the
graph honest.)

Weight kept at 0.05 deliberately: under normalisation that is roughly the
pressure NEAR rays already survived in the failed run. The failure was
far-field domination, not near-field pressure, so lowering the weight uniformly
would have weakened the term exactly where it is wanted.

## CORRECTION: the dominant near-field defect is BLUR, not transparency

A screenshot of the merged preview showed the bottom third as an opaque smeared
mass -- not see-through. The hole metric this whole investigation was gated on
measures whether a ray accumulates enough ALPHA. It says nothing about whether
the ground carries TEXTURE. Both defects are real; the second is what dominates
the image, and the depth-distortion loss does not address it at all.

`scripts/sharpness.py` -- mean |grad| in the bottom third, render vs ground
truth, from a Brush eval dir:

| iter | aug2 bottom-third detail retained | top half |
|---|---|---|
| 5000 | 12.6% | - |
| 10000 | 14.9% | 36.7% |
| 20000 | 18.2% | - |
| 30000 | **22.6%** | 42.0% |

The near field retains under a quarter of the real texture at full convergence,
roughly half what the rest of the frame retains.

**It is not the viewer's splat cap.** ply2splat ranks by opacity x projected
area and keeps the top 3M of 8M, and ground splats are low-opacity and small,
so culling them was the obvious suspect. Measured across three caps, bottom-third
detail is 0.00222 / 0.00225 / 0.00225 at 1.5M / 3M / 6M -- flat. The detail was
never reconstructed; the viewer is not discarding it.

### Training stops exactly where detail is improving fastest

Increments per 10k iterations: **+3.3** (10k->20k), **+4.4** (20k->30k). The
curve is ACCELERATING at the point every run so far has stopped. Nothing in this
project has ever trained past 30k.

`scripts/chain_overnight.sh` runs merged at **60k** to test it. That is a
genuinely different run rather than "more of the same", because
`--total-train-iters` also sets the LR decay schedule.

Merged is blurrier than single-site at equal iterations (10.4% vs 14.9% at 10k):
the same 8M budget spread over twice the area buys coverage, not fidelity.

### The merged hole metric is INVALID -- terrain_grid cannot span two sites

`measure.py` on the merged preview reported LOW opacity 0.966 / holes 1.2%
against aug2's 0.654 / 33.5%, which would have read as wonderful news. It is an
artefact.

`terrain_grid` fits ONE height field over the whole extent. The merged scene is
two DISJOINT sites with empty space between them, so the field interpolates
across the gap and AGL comes out meaningless:

    level-ish cameras          521 of 7470
    AGL percentiles [5,20,50,80,95]   -0.196  -0.044  -0.001  0.231  0.328
    cameras computing as BELOW terrain     52.0%

Median AGL of -0.001 and half the drone underground. The "low altitude" cohort
is therefore selected at random, and every merged number from `measure.py`,
`ground_thick.py` and `dilate_low.py` is meaningless. Single-site numbers are
unaffected.

Fix before trusting any merged slab metric: fit the terrain field per CLIP (or
per connected component of camera positions) instead of globally.

# ============================================================
# THE ACTUAL CAUSE: the viewer's near plane. One line.
# ============================================================

The see-through ground was **not** primarily a reconstruction defect. It was
`const znear = 0.2` in `viewer/main.js`.

Splats are frustum-culled on their CENTRE depth. Median flight AGL on these
scenes is ~0.05-0.16 in viewer units, so `znear = 0.2` is roughly **4x the
drone's lowest altitude** -- the ground directly beneath a low-flying camera
sits INSIDE the near plane and is culled outright. The bottom third of the frame
then shows the background through it.

Same splat file, same cameras, only `znear` changed:

| scene | znear = 0.2 (shipped) | znear = 0.01 |
|---|---|---|
| aug2 baseline, LOW cohort | opacity 0.621, holes **35.6%** (worst 77.9%) | opacity 0.949, holes **0.9%** (worst 4.0%) |
| merged cam 2500 | 0.645, 35.9% | 0.974, 1.0% |
| merged cam 7448 | 0.729, 28.6% | 0.999, 0.0% |
| merged cam 924 | 0.747, 17.6% | 0.972, 0.0% |

Matched good poses: unchanged or slightly better, never worse. The viewer has NO
depth buffer -- it sorts and alpha-blends -- so `znear` buys nothing here and
costs no precision to shrink. Decomposition: the frustum CULL dominates, the
`clamp(ndc_z+1,0,1)` near-fade adds 1.6-8.5 points on top, and lowering znear
fixes both.

## Why this was missed for so long, twice

`znear` was tested EARLY and recorded as a red herring -- "lowering znear to 0.01
moved bottom-third opacity by <0.03". That measurement was taken over a COHORT
MEAN. The effect exists only at the lowest-altitude poses, and averaging over a
cohort that is mostly fine dilutes a 36-point effect into nothing.

Every metric in this investigation had the same shape. `measure.py` reports a
cohort mean. `ground_thick.py` aggregates over 300 columns. The defect lives in
the **tail** -- about 7% of poses exceed 20% holes -- and every tool was built to
report the middle. The user could see it immediately because they were flying
through the bad poses, not averaging them.

Two other measurements pointed straight at it and were misread:

* `corr(alpha, splats/px) = +0.94` but `corr(alpha, splats within 0.3 of camera)
  = -0.46`. The geometry was PRESENT and not being RENDERED. That sign flip is
  only explicable by culling, and it was visible before the znear sweep was run.
* `corr(alpha, AGL) = +0.35` -- lower flight, worse holes -- which is exactly the
  signature of a fixed near plane, not of a diffuse slab.

## What this invalidates

The 33.5% hole figure that motivated the slab diagnosis, `--flatten-strength`
(3a), the depth-distortion kernel (3b) and the opacity boost was overwhelmingly
this bug. With `znear = 0.01` the SAME baseline PLY measures 0.9% holes.

Still standing, because they were measured on splat geometry rather than through
the viewer: the ground IS a volumetric slab (47-49% of opacity mass below the
visible surface, 10->90% transition 131-144% of flight altitude). That is real
and unchanged. It simply was not what made the bottom third see-through.

Still standing as a genuine open defect: **near-field BLUR**. Bottom-third detail
retention is 22.6% at 30k on aug2 and 14.8% on merged, and the curve is still
accelerating at 30k. That is unaffected by znear and is now the main remaining
quality issue.

The opacity boost is no longer needed as a workaround -- unboosted with the fix
is 4.4% worst-case versus 51.2% before -- so the shipped merged build is now the
faithful, unboosted reconstruction.

---

## The znear fix, checked on every scene

The one-line change (`viewer/main.js`, `znear 0.2 -> 0.01`) was found on the
merged scene. This section verifies it on all three, on the exact assets that
are deployed, and states what it does NOT fix.

### Two selectors, because AGL is not available everywhere

`scripts/znear_audit.py` picks the lowest-AGL level-looking cameras. Valid on a
single-site scene only -- `terrain_grid` fits ONE height field, so on merged it
interpolates across the empty space between hilltop and cemetery and every AGL
it returns is meaningless.

`scripts/znear_band.py` needs no terrain model, so it works on merged too. It
ranks cameras by the *mechanism*. The viewer rejects a splat when
`pz < -margin*z`, which reduces to a pure near-depth cut at

    z < znear*k / (margin + k),    k = zfar/(zfar-znear)

so with `zfar=200, margin=1.2` the cut sits at **z < 0.0910** for `znear=0.2`
and **z < 0.0045** for `znear=0.01`. Splats between those two depths are exactly
what the fix recovers. Note the shipped cull was never at 0.2 -- the 1.2 margin
divides it by ~2.2 -- but 0.091 is still well inside the flight envelope.

### Results (bottom third of frame, holes = pixels below 0.5 alpha)

| scene | cohort | n | znear 0.2 | znear 0.01 | worst | regressed |
|---|---|---|---|---|---|---|
| aug2 cemetery 3M | banded WORST | 6 | 74.3% | **0.2%** | 90.7 -> 0.9 | 0/6 |
| aug2 cemetery 3M | banded CTRL | 4 | 19.6% | **1.3%** | 42.8 -> 4.5 | 0/4 |
| aug2 cemetery 1.5M | AGL LOW | 8 | 29.2% | **4.7%** | 78.1 -> 26.0 | 0/8 |
| aug2 cemetery 1.5M | AGL GOOD | 5 | 0.6% | 0.5% | 1.7 -> 1.7 | 0/5 |
| merged 3M | banded WORST | 6 | 20.1% | **0.4%** | 68.8 -> 2.6 | 0/6 |
| merged 3M | banded CTRL | 4 | 15.6% | **0.1%** | 39.0 -> 0.2 | 0/4 |
| aug hilltop 3M | banded WORST | 6 | 13.3% | **0.6%** | 36.9 -> 3.5 | 0/6 |
| aug hilltop 3M | banded CTRL | 4 | 0.4% | **0.0%** | 1.6 -> 0.2 | 0/4 |
| aug hilltop 1.5M | AGL LOW | 8 | 4.9% | **1.1%** | 19.2 -> 5.5 | 0/8 |
| aug hilltop 1.5M | AGL GOOD | 5 | 4.4% | **0.4%** | 15.6 -> 1.6 | 0/5 |

**Not one pose out of 56 got worse.**

### Three things this measurement changed

**It is not a tail-only effect.** The CONTROL cohorts -- cameras at the median
of the banded ranking, i.e. deliberately unremarkable poses -- improve almost as
much as the worst ones (merged 15.6% -> 0.1%, aug2 19.6% -> 1.3%). The tail is
where it was most visible, not where it lived. The original "<0.03 opacity
change" null was wrong about the size of the effect, not just its distribution.

**Severity is set by the scene's normalisation, not by the flying.** `znear` is
in normalised viewer units, and `ply2splat.py` scales each scene so the camera
p90 radius lands at 3.0. That gave aug2 a median AGL of 0.24 and the hilltop
0.47 -- so the same drone, same day, same pilot produced one scene hit twice as
hard as the other. This also means the bug's severity was never a property of
the footage, and no amount of looking at the reconstruction would have found it.

**The banded share does not predict severity well.** aug2's worst pose has 5.93%
of its frustum splats in the cull band; merged's has 1.42%, yet merged CTRL
poses with 0.08% still lost 39% of the bottom third. What matters is not how
many splats are culled but whether the culled ones are the ground you are
looking at. Use the band to *find candidates*, then render.

### What the fix does not do

`znear` is a rendering bug. It changes nothing about the reconstruction:

- the ground slab is still real (47-49% of opacity mass below the visible
  surface, transition thickness 131-144% of flight altitude)
- near-field BLUR is untouched and is now the main remaining defect -- detail
  retention 22.6% @30k aug2, 14.8% merged, and the curve was still accelerating
  at 30k (+3.3 pts 10k->20k, +4.4 pts 20k->30k)
- residual holes survive at a few poses even after the fix (aug2 1.5M worst is
  still 26.0%), so there is genuine missing near-field geometry underneath

The opacity boost (`--ground-opacity-boost`) was a workaround for this bug and
has been removed from all deployed assets. All three scenes are now built on
identical terms: 3M splats, unboosted, up derived from ground geometry and
validated with `check_up.py`, cameras and trajectory regenerated together.
Rebuild any of them with `scripts/rebuild_scene.sh <model> <ply> <prefix>`.
