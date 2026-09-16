# FPV Gaussian Splatting — Building 3D Reconstructions from Drone Footage on a Mac

A source document for a portfolio write-up. Everything below is measured, not
estimated; where a number is an estimate it says so. Negative results are
included deliberately — most of the interesting work in this project was
disproving my own hypotheses.

---

## 1. One-paragraph summary

I flew a DJI O4 Pro FPV drone over two sites, then built photorealistic 3D
Gaussian Splatting reconstructions from the raw footage — entirely on an Apple
Silicon Mac, with no CUDA. The pipeline goes video → gyro-driven frame selection
→ COLMAP structure-from-motion → Brush (Rust/WebGPU 3DGS trainer) → a custom
WebGL viewer with a "ghost replay" feature that flies the original drone
trajectories through the reconstruction alongside the source footage. Along the
way I implemented a 2DGS depth-distortion loss in Rust/CubeCL from the paper,
merged two flight sessions 6 days apart into a single coordinate frame, and
spent most of the effort chasing a visual defect that turned out to be a
one-line bug in the *viewer*, not the reconstruction.

**The headline engineering lesson:** the defect that drove weeks of work —
"the ground looks see-through when flying low" — was `const znear = 0.2` in the
viewer. Four separate reconstruction-side interventions were built and measured
against a symptom that a rendering bug was causing.

---

## 2. Constraints that shaped everything

| Constraint | Consequence |
|---|---|
| **No CUDA** — Apple Silicon only (M5 Pro, 48 GB unified) | Ruled out every mainstream 3DGS implementation. Used **Brush**, a Rust/WebGPU trainer, on the Metal backend. COLMAP ran CPU-only. |
| **Consumer FPV footage, not a capture rig** | Fast motion, rolling shutter, wide fisheye lens, no turntable or controlled orbit. Camera pitched *up* 15–29°, so the ground is seen at grazing angles. |
| **Two sessions, 6 days apart** | Different lighting, different vegetation state. Merging them required proving cross-session feature matches actually existed before committing 18 hours of COLMAP. |
| **4 GiB GPU single-buffer limit** | Hard ceiling of ~23.7M splats, unrelated to the 48 GB of system RAM. Discovered by crashing into it. |
| **Everything runs locally** | No cloud GPU. A single training run is 5–28 hours, which makes experiment design (what to measure, what to ablate) the real bottleneck, not compute. |

---

## 3. The pipeline

```
DJI O4 Pro (2688×2016, 50fps, H.264)
  │
  ├─ gyro extraction ────────────────► per-frame quaternions
  │                                     (DJI embeds IMU data; only valid at FOV:Normal)
  ├─ adaptive frame selection ───────► scripts/select_frames2.py
  │     keep a frame when cumulative rotation ≥ 8°, bounded by kmin/kmax
  │     so translation-dominated flight still samples densely
  │
  ├─ frame extraction (1920×1440) ──► frames/<scene>/<clip>_<frameindex>.jpg
  │
  ├─ COLMAP SfM (CPU) ──────────────► exhaustive+sequential matching,
  │     incremental mapper, global BA   sparse/0/{cameras,images,points3D}.bin
  │
  ├─ Brush 3DGS training (Metal) ───► site_<iter>.ply  (8M–22M splats)
  │
  ├─ ply2splat.py ──────────────────► .splat  (antimatter15 32-byte format)
  │     levels the scene from a derived up-vector, normalises scale,
  │     caps splat count by opacity × projected area
  │
  └─ WebGL viewer ──────────────────► sorted alpha-blended splats + ghost replay
```

### Reconstruction quality achieved

| | hilltop (aug) | cemetery (aug2) | **merged** |
|---|---|---|---|
| source clips | 0050–0055 | 0057–0060 | all 9 |
| frames registered | 3,777 | 3,693 | **7,470 / 7,471** |
| 3D points | — | — | **3.36 M** |
| observations | — | — | **22.1 M** |
| mean reprojection error | — | — | **0.821 px** |
| COLMAP wall time | — | — | **1,076 min** |

The merge produced **one** model, not two disconnected components — the failure
mode I was most worried about. 7,470 of 7,471 frames registered.

---

## 4. What I built

### 4.1 A depth-distortion loss for Brush (Rust / CubeCL / Burn)

Brush ships without any surface regulariser. 3DGS has no prior that makes a
surface *thin* — from grazing views, a thick semi-transparent slab and a thin
opaque surface produce an identical image and therefore an identical loss. So I
implemented the 2DGS depth-distortion loss:

```
L = Σ_{i,j} w_i w_j |t_i − t_j|
```

Naively O(N²) per ray. Rewritten in O(N) prefix-sum form:

```
L = Σ_i w_i (t_i·A_i − B_i)     where  A_i = Σ_{j<i} w_j ,  B_i = Σ_{j<i} w_j t_j
```

The interesting part is the **backward pass**. The gradient needs a *suffix* sum
as well as a prefix sum, and the backward rasterizer walks front-to-back with no
second pass available. Euler's theorem supplies it for free: `L` is homogeneous
of degree 2 in `w`, so

```
R_0 = Σ_i w_i g_i = 2L
```

which means the suffix sum can be seeded from the forward pass's own output and
walked down exactly the way the kernel already walks remaining colour. Alpha
also reaches the loss by **two** routes — directly via `w_i = α_i·T_i`, and
indirectly because every `T_j` for `j > i` carries a `(1−α_i)` factor — and both
have to be accumulated.

**Files:** `bwd/kernels/rasterize_backwards.rs`, `bwd/kernels/project_backwards.rs`,
`gaussian_splats.rs`, `bwd/{render_bwd,burn_glue}.rs`, `brush-train/src/{config,train}.rs`

**Validation:** finite-difference tests (`brush-bench-test/tests/distortion_backward.rs`),
including an anti-vacuous guard asserting the gradient is *non-zero* (a
finite-difference test passes trivially if both sides are zero) and a test
asserting the loss does not touch SH colour gradients at all.

**Two traps worth recording:**
- The `v_combined` gradient buffer stride was changed from 10 to 11
  **unconditionally**, not conditionally on the loss being enabled. A stride
  that differs between two kernels is a silent-corruption bug, not a crash.
- `project_backwards` had an early-out `if !any_grad { return }`. Depth-only
  gradients would have been silently dropped without adding
  `|| v_depth_in != 0.0` to that condition.

**Result: a measured negative.** At `--distortion-weight 0.05` unnormalised, the
loss collapsed the scene's depth range (splat radius p99 fell 17.45 → 7.78) and
PSNR dropped 22.14 → 19.80 dB. With per-ray normalisation it no longer collapsed
but still degraded detail (16.9% vs the 22.6% baseline). The implementation is
correct; the intervention doesn't help this scene.

### 4.2 Ghost replay in the viewer

The differentiating feature. The viewer reconstructs the transform that
`ply2splat.py` applied (recovered from the camera set, with a residual of
`3e-14`), then flies the original drone trajectories through the reconstruction
as coloured "ghosts" — with the source video playing in sync in a corner, at the
same pose. It supports onboard/chase cameras, trails, scrubbing, speed control,
and an undistort toggle.

Two non-obvious problems:
- **Time comes from the frame index, not the file order.** Frame filenames
  encode the source video frame number, which is the only reliable clock.
- **The blend operator forbids the obvious draw order.** Splats are drawn
  back-to-front with premultiplied alpha; adding opaque geometry (the drone
  models, trails) on top breaks the blend. Solution: a depth prepass, then
  compositing the geometry *under* the splat pass.

### 4.3 Measurement tooling

The project produced ~90 analysis scripts. The ones that mattered:

| script | what it measures |
|---|---|
| `sharpness.py` | high-frequency energy (mean \|∇\|) in the render vs ground truth, by region — **texture** |
| `asset_holes.py` | bottom-third opacity and hole fraction at low poses — **coverage** |
| `detail_stratified.py` | detail retention binned by *local GT texture*, near vs far at matched difficulty |
| `screen_size.py` | reproduces Brush's own `max_screen_size` statistic from a PLY + COLMAP cameras |
| `znear_band.py` | ranks cameras by the frustum-cull mechanism; needs no terrain model, so valid on multi-site scenes |
| `check_up.py` | validates a candidate up-vector by asserting cameras sit *above* terrain |
| `coverage_probe.py` | correlates alpha against splat density, camera density, and altitude |

---

## 5. The main investigation: "the ground is see-through"

This is the spine of the project and the most transferable part.

### 5.1 The symptom

Flying low in the viewer, the bottom third of the frame showed the background
*through* the ground. Reported repeatedly, and it made the reconstruction feel
broken regardless of the metrics.

### 5.2 Four reconstruction-side hypotheses, all built, all measured

**(a) The ground is a slab, not a surface.** Measured with `ground_thick.py`:
47–49% of the ground's opacity mass sits *below* the visible surface; the
10→90% opacity transition spans 131–156% of median flight altitude; ground splat
opacity median 0.26 with 83% under 0.5; the thin axis sits a median **47° off**
the surface normal.

This is all true. It is also *not what made the ground see-through.*

**(b) A flatness prior.** Shrink each splat's scale axes toward a disc at every
refine step. **Made it worse** — holes 33.5% → 38.9%, post-hoc flattening
35.9% → 61.3%. The reason generalises: a ground splat's thin axis is 1.0% of
flight altitude while the slab is 125%, so the slab is **~129× thicker than one
splat**. It is a fact about *where splat centres are*, not about splat shape.
Any intervention that only rescales axes is aimed at the wrong quantity — a
one-line check that rules out an entire class of fix.

**(c) The depth-distortion loss.** See §4.1. Negative.

**(d) A post-hoc opacity boost.** `a → 1−(1−a)^k` multiplies optical depth by
exactly `k` in every direction. It worked as a mitigation but was a workaround,
and was removed once the real cause was found.

### 5.3 The actual cause: one line in the viewer

```js
const znear = 0.2;   // viewer/main.js
```

The viewer frustum-culls each splat on its **centre depth**. Median flight
altitude in normalised viewer units is ~0.05–0.16 — so the near plane sat at
roughly **4× the drone's lowest altitude**, and the ground directly beneath a
low-flying camera was culled outright. The viewer has no depth buffer (it sorts
and alpha-blends), so `znear` bought nothing and cost nothing to shrink.

Same splats, same cameras, only `znear` changed:

| scene | cohort | znear 0.2 | znear 0.01 |
|---|---|---|---|
| cemetery 3M | worst 6 poses | 74.3% holes | **0.2%** |
| cemetery 3M | control 4 poses | 19.6% | **1.3%** |
| merged 3M | worst 6 poses | 20.1% | **0.4%** |
| merged 3M | control 4 poses | 15.6% | **0.1%** |
| hilltop 3M | worst 6 poses | 13.3% | **0.6%** |

**0 of 56 poses got worse.**

### 5.4 Why it hid for so long — the methodological lesson

I tested `znear` *early* and recorded it as a red herring: "lowering znear to
0.01 moved bottom-third opacity by <0.03." That was a **cohort mean**. The
effect only exists at the lowest-altitude poses; averaging over a mostly-fine
cohort diluted a 36-point effect into nothing.

Every tool I had built reported a middle. The defect lived in the tail. The
person flying the viewer saw it instantly because they fly *through* the bad
poses instead of averaging over them.

Two clues pointed straight at it and were misread:
- `corr(alpha, splats/px) = +0.94` but `corr(alpha, splats within 0.3 of camera) = −0.46`
  — geometry **present but not rendered**, which only culling explains.
- `corr(alpha, altitude) = +0.35` — the signature of a fixed near plane, not a
  diffuse slab.

And one more thing that only became clear afterwards: **severity was set by the
scene's normalisation, not by the flying.** `ply2splat.py` scales each scene so
the camera p90 radius lands at 3.0, which gave the cemetery a median altitude of
0.24 and the hilltop 0.47 — the same drone, same day, one scene hit twice as
hard. No amount of looking at the reconstruction would have found that.

---

## 6. The second investigation: near-field blur

With transparency fixed, a real second defect remained: near-field ground is
*blurry*. Different failure mode, different cause, and the obvious metric
conflates it with something else.

### 6.1 The metric was half wrong

`sharpness.py` divides render high-frequency energy by GT high-frequency energy
per region. Bottom third scored 18%, top half 40% — which reads as a
catastrophic near-field defect. But the bottom third's GT carries **2.44×** the
high-frequency energy (close grass vs sky and distant hills). The regions were
never comparable.

`detail_stratified.py` bins 8×8 blocks by local GT texture and compares near vs
far *at matched difficulty*:

| GT texture band | near | far | near/far |
|---|---|---|---|
| 0.0146–0.0331 | 20.8% | 45.0% | 0.46 |
| 0.0331–0.0538 | 18.5% | 40.2% | 0.46 |
| 0.0538–0.0685 | 17.5% | 38.5% | 0.45 |
| 0.0685–0.0845 | 15.9% | 37.2% | 0.43 |
| 0.0845–0.2245 | 14.7% | 33.7% | 0.44 |

So the defect is real — at identical content difficulty the near field retains
under half what the far field does — but roughly half the apparent gap was just
harder content. And a **flat ratio across every texture band** is the signature
of a *scale* problem, not a content one.

### 6.2 A controlled ablation

Five training runs on the cemetery scene, all 20k iterations, all sharing one LR
schedule so they rank fairly:

| run | config | bottom third | near/far ratio | verdict |
|---|---|---|---|---|
| C0 | 8M splats | 18.2% | 0.45 | control |
| P1 | 16M cap | 18.6% | 0.45 | **capacity does nothing** |
| **P2** | **16M + `--split-at-screen-size 0.03`** | **20.3%** | **0.50–0.61** | **the lever** |
| P4 | native 2688 resolution + split | 20.2% | 0.48–0.59 | tie — resolution is not the lever |
| P5R | 22M + split | 20.5% | 0.50–0.61 | +0.2, capacity exhausted |
| P3 | + LPIPS (cropped) | **14.7%** | 0.44 | actively harmful |

**P2 is the only intervention that moved the ratio** — the near-field-specific
deficit — and near-field gained roughly twice what far-field did.

### 6.3 Three findings from that ablation

**Saturation proves a cap binds; it says nothing about whether relieving it
helps.** Every run ever done exported *exactly* 8,000,000 splats, already by
iteration 10,000. That looked like overwhelming evidence the cap was the
constraint. Given a 16M budget the optimiser took **8.67M** by 10k. The real gap
was 8.0 vs 8.67M, not 8 vs 16M.

**`--split-at-screen-size` was a no-op, and is coupled to the cap.** Default
0.5 — but `screen_size.py` measures p99 = 0.103 and only **0.01%** of splats
above 0.5, five times past p99.9. Nothing was ever force-split. Worse, its
splits are funded from `max_splats − current`, which had been **zero** since
iteration 10k. Lowering the threshold alone would also have done nothing. The
two flags only work together.

**Resolution is not the lever, and the metric lies about it by default.**
Training at native 2688 scored 20.2% once downsampled to a common yardstick — a
dead tie — and 16.1% against its own 2688 GT. It failed *proportionally* at a
harder target rather than resolving more. `sharpness.py` originally credited it
19.7% because it picked the first alphabetical GT directory (the 1920 set) and
upscaled it to the 2688 render, scoring against a reference with no detail above
1920's Nyquist. **When comparing runs at different resolutions, bring them to one
yardstick first.**

---

## 7. The final run — and the mistake that undid it

The winning config went to a 60,000-iteration run on the merged scene: 28 hours,
16M splats, all 7,470 cameras. Texture improved exactly as predicted:

| | old (8M, 30k) | new (16M + split, 60k) |
|---|---|---|
| bottom third detail | 14.8% | **16.6%** |
| top half | 34.7% | 35.6% |

Near-field improved in **every** texture band, and the near/far ratio rose in
five of six.

**And it looked visibly worse.** The ground was see-through again.

The cause was the export step, not the training. `ply2splat.py` keeps the top N
splats by opacity × projected area:

- old run: 8M → 3M = **keeps 37.5%**
- new run: 16M → 3M = **keeps 18.8%**

Doubling the trained splat count while holding the export cap fixed throws away
four fifths instead of two thirds — and `--split-at-screen-size` had made each
survivor *smaller*. Both effects cut coverage.

| asset | splats | bottom-third opacity | holes | worst pose |
|---|---|---|---|---|
| old 8M/30k → 3M | 3,000,000 | 0.976 | 0.1% | 0.4% |
| new 60k → 3M | 3,000,000 | 0.878 | **5.6%** | **18.1%** |
| new 60k → 8M | 8,000,000 | 0.949 | 0.3% | 1.7% |

Raising the export cap recovered essentially all of it — no retraining needed.

**This was the same metric mistake twice in one project.** `sharpness.py`
measures *texture*; hole fraction measures whether a ray accumulates *alpha*.
A 28-hour run was gated on the first without re-checking the second — the exact
confusion that earlier made a blur defect look like a transparency defect.

---

## 8. Complete table of negative results

Kept deliberately: the disproofs took more effort than the successes.

| intervention | measured outcome |
|---|---|
| Flatness prior (training-time) | holes 33.5% → 38.9% — worse |
| Flattening splats post-hoc | holes 35.9% → 61.3% — much worse |
| Depth-distortion loss, w=0.05 | scene collapse; radius p99 17.45 → 7.78; PSNR 22.14 → 19.80 dB |
| Depth-distortion loss, normalised | detail 22.6% → 16.9% — worse |
| `cov_blur` screen-space dilation | holes 30.1 / 30.1 / 30.0% at dilate 0 / 0.3 / 1.0 — null at 3.3× the real value |
| Viewer splat cap (at fixed training) | detail flat at 1.5M / 3M / 6M |
| Raising `max_splats` 8M → 16M | +0.4 detail, near/far ratio unmoved |
| Native 2688 training resolution | dead tie with 1920 on a common yardstick |
| LPIPS perceptual loss | 14.7% vs 18.2% control — actively harmful |
| Post-hoc opacity boost | worked, but was a workaround for the viewer bug |

---

## 9. Engineering war stories worth telling

**LPIPS was broken, then unaffordable, then harmful.** `--lpips-loss-weight`
panicked 4 seconds into training at `burn-dispatch/src/tensor.rs:98`
`unreachable!()`. The Burn fork carries autodiff-ness at *runtime* in a
`BackendTensor` enum rather than in the type, so weights loaded from a record are
plain `Float` variants and the first op mixing them with autodiff activations
calls `.autodiff()` on a `Float`. Fixed with a `ModuleMapper` lifting each
parameter through `Tensor::from_inner`. Then it ran — at **29 s/iter** against
~0.5 for the entire rest of the step, i.e. a 20k run would have taken months.
Added `--lpips-crop` to evaluate a random 256px window instead of the full
frame. Then it turned out to hurt quality anyway.

**A 4 GiB single-buffer ceiling, not a RAM ceiling.** A 24M-splat run died with
`failed to reserve 4336857088 bytes`. That is just past 4 GiB; the per-splat
buffer is ~181 bytes, so the hard ceiling is **~23.7M splats** — on a machine
with 48 GB of RAM.

**A `pgrep` waiter that lied.** `until pgrep -f 'a\|b'; do sleep; done` exits
*immediately* and prints its success message: `pgrep -f` takes an *extended*
regex where alternation is bare `|`, so `\|` matched a literal pipe, matched
nothing, and the loop's exit condition was true on the first iteration. It
reported "ALL SCANS DONE" while three renders were still running. A clean exit
is not evidence.

**Editing a running bash script.** Bash reads scripts incrementally by file
offset, so editing a script mid-run can resume execution at a wrong offset.
Every chain script in this project was killed by PID, edited, syntax-checked with
`bash -n`, and relaunched — never edited in place.

**A silent-scoring bug that read as a result.** `eval_compare.py` had hardcoded
ground-truth paths, so on any newer frameset it reported `images: 0` — which
reads as a finding rather than an error.

**`terrain_grid` breaks silently on multi-site scenes.** It fits *one* height
field, so on the merged hilltop+cemetery scene it interpolated across the empty
space between sites: median camera altitude −0.001 and **52% of cameras
computing as underground**. It reported a flattering 1.2% hole figure that
nearly got believed.

**Brush's `Vertical axis` PLY comment has Z negated.** A wrong up-vector puts
every camera underground while still looking plausible. Every scene's up is now
derived from ground geometry and validated by asserting cameras sit above
terrain (`check_up.py`) — for the merged scene, the correct sign gives 1% of
cameras below terrain and the wrong one gives 99%.

---

## 10. Final deliverable

A merged reconstruction of two sites from 9 clips and 7,470 registered frames,
viewable in a browser with the original flight paths replayable through it
alongside the source footage.

- **3,000,000 splats** exported from an 8M-splat 30k-iteration training run
- **0.1% mean hole fraction** in the bottom third at low-flight poses (worst pose 0.4%)
- **0.821 px** mean reprojection error across 22.1M observations
- ~67 fps in the browser

---

## 11. Tech stack

**Reconstruction:** COLMAP 4.1.1 (CPU), Brush (Rust / WebGPU / Metal), Burn,
CubeCL
**Analysis:** Python, NumPy, OpenCV, Pillow — including a from-scratch CPU
splat rasteriser (`raster.py`) that reproduces the viewer's projection,
covariance, half-float packing and blend maths exactly, so rendering hypotheses
could be A/B'd without a GPU
**Viewer:** WebGL2, antimatter15 splat viewer heavily extended
**Infra:** Cloudflare quick tunnels for sharing, Synology NAS for archival,
bash orchestration chains with GPU-serialised job queues

---

## 12. What a reader should take away

1. **A cohort mean can hide a catastrophic defect in the tail.** The bug that
   drove the entire project was measured early, averaged away, and written off.
2. **Two metrics that both sound like "quality" can move in opposite
   directions.** Texture and coverage are different failure modes; optimising
   one silently wrecked the other, twice.
3. **Saturation is not causation.** Every run pinning exactly at the splat cap
   looked like overwhelming evidence the cap was binding. Lifting it changed
   almost nothing.
4. **Check the cheap ruling-out calculation first.** The slab was 129× thicker
   than a single splat — one line of arithmetic that invalidated an entire
   category of fix before any of it was built.
5. **Negative results need the same rigour as positive ones**, or you cannot
   trust them when they contradict your next idea.
