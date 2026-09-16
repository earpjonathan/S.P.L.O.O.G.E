# 3b — depth-distortion loss for Brush (scoped, not yet implemented)

## Why this is the correct loss for our defect

Measured on both terrain scenes: ~48% of the ground's opacity mass sits *below*
the visible surface and the 10->90% opacity transition spans 131-144% of flight
altitude. The 2DGS depth-distortion loss is, literally, "concentrate each ray's
contribution at one depth":

    L_d = sum_{i,j} w_i w_j |t_i - t_j|          w_i = alpha_i * T_i

It is the only term that penalises the *spread* of weights along a ray, which
is exactly the quantity that is wrong. `--flatten-strength` (3a) shapes
individual splats; this shapes their arrangement along the ray. They are
complementary, not alternatives.

## O(N) formulation (what actually goes in the kernel)

Front-to-back, with `t` ascending, `|t_i - t_j| = t_i - t_j` for `j < i`:

    A_i = sum_{j<i} w_j          B_i = sum_{j<i} w_j t_j
    L_d = sum_i w_i * (t_i * A_i - B_i)

Two running scalars per pixel. No extra passes, no sorting (the rasteriser is
already depth-ordered).

## Architecture (verified against the source, not assumed)

`prep.finish(state, output.out_img)` in `bwd/burn_glue.rs:372` gives this
autodiff node **exactly one differentiable output**, and the backward starts
from `v_output = grads.consume::<B>(&ops.node)`. So per-pixel distortion cannot
simply be a second returned tensor.

**Resolution: carry distortion as an extra CHANNEL of `out_img`.**
`render.rs:248` already sets `out_dim = if bwd_info { 4 } else { 1 }`, so the
training path owns its own channel count; making it 5 when the distortion flag
is on is a contained change and gradients then flow through the existing
`v_output` with **no new autodiff plumbing at all**. The inference path (packed
u8x4) is untouched and never pays for this.

Two consequences worth banking:
* The loss module slices `img[..., 0..4]` for photometric and `img[..., 4]` for
  distortion. Every other `out_img` consumer must be audited for a hardcoded 4.
* `finite_diff_weighted_loss` already multiplies `diff.img` by a random
  `[h, w, c]` weight tensor, so widening `c` exercises the new channel's
  gradient nearly for free. That is the cheapest possible first test.

## Work items

1. ~~**`kernels/helpers.rs`** — widen the payload to carry view depth.~~
   **DONE.** `PROJECTED_LANES` 9 -> 10, `Splat` gained a `depth` field,
   `project_visible.rs` sets `depth: mean_c.z()`. The tile copy loop in
   `rasterize.rs` iterates `0..PROJECTED_LANES_USIZE`, so shared-memory sizing
   and the copy widened automatically (+1 KB of threadgroup memory, well inside
   Metal's 32 KB). No behaviour change yet -- the lane is written and read but
   unused, which is deliberate: it builds and runs identically, so the plumbing
   is verified before any math depends on it.

2. ~~**`kernels/rasterize.rs`** — forward accumulation.~~ **DONE and TESTED.**
   `a_acc`/`b_acc` registers in the contributing branch, pair term formed
   BEFORE the prefix sums update, written to channels 4/5/6 under the
   `RasterPass::BackwardDistortion` comptime flag (`out_dim` 7). Inference path
   byte-identical.

   `crates/brush-bench-test/tests/distortion_forward.rs` -- 5 tests, all green:

   * **`a_total_equals_alpha`** is the load-bearing one. `sum_i w_i` with
     `w_i = alpha_i T_i` telescopes to `1 - T_final`, which channel 3 already
     holds, so channels 3 and 5 must agree to floating point. Accumulating
     outside the contribute branch, or after the transmittance update, fails
     this immediately and unambiguously.
   * `distortion_is_non_negative` pins the ORDERING: front-to-back traversal
     makes every pair term `w_i (t_i - t_j)` with `t_i >= t_j`, so a negative
     value means the prefix sums were updated before the pair term was formed.
   * `single_splat_has_zero_distortion` (no pairs),
     `stacked_splats_have_positive_distortion` (the slab case),
     `out_img_has_seven_channels` (plumbing).

   None of these need gradients. That is deliberate: with the forward proven,
   any finite-difference failure later is unambiguously a BACKWARD bug.

3. ~~**`bwd/kernels/rasterize_backwards.rs`** — the backward.~~ **DONE and
   FINITE-DIFF VERIFIED.** The derivation below held up unchanged; nothing in
   it needed correcting once implemented.

   ### Gradients (as implemented)

   With `L = sum_i w_i (t_i A_i - B_i)`:

       dL/dw_k = (t_k A_k - B_k) + (SB_k - t_k SA_k)
       dL/dt_k = w_k (A_k - SA_k)
       SA_k = A_total - A_k - w_k        SB_k = B_total - B_k - w_k t_k

   The alpha chain has two routes, since `w_i = alpha_i T_i` and every `T_i`
   with `i > k` carries a `(1 - alpha_k)` factor:

       R_0 = sum_i w_i g_i = 2L      (Euler; L is homogeneous of degree 2 in w)
       R_{k+1} = R_k - w_k g_k
       v_alpha_eff += v_dist * (g_k * T_k - R_{k+1} * ra)

   `dL/dt_k` reaches the mean as a pure +z push in camera space, added to
   `v_mean_c` before the rotation back to world.

   ### What the implementation actually required

   * `pix_state` 4 -> 7 floats, comptime-strided so the default training path
     keeps its 4 KiB and its occupancy. Same for `pix_base`, which had to stop
     being `pix_id * 4` -- the trap this plan called out.
   * `v_combined` 10 -> 11 lanes, **unconditionally**. Constant stride was
     chosen over a comptime one deliberately: a stride that varies between
     two kernels is exactly the silent-corruption failure this plan warned
     about, and the cost is one f32 per visible splat. The atomic WRITE to
     lane 10 is still comptime-gated, so the default path pays nothing; the
     read side is a no-op because the buffer is zero-initialised.
   * `v_depth_in` had to join the `any_grad` early-out in
     `project_backwards.rs`. Without it a splat whose ONLY gradient is depth
     returns before writing anything -- silently, and invisible to any test
     that does not isolate the distortion channel.
   * New `RasterPass::BackwardDistortionSmoothCutoff`. Distortion and the C^1
     cutoff are orthogonal concerns, and finite-diff needs both at once;
     `BackwardDistortion` alone uses the hard 1/255 step, which central
     differences straddle.

   ### Depth normalisation: measured, not needed

   `t_k A_k - B_k` is a difference of similar large numbers, so f32
   cancellation scales with depth. Measured on the actual scene
   (`work/eval_base8M_30k.splat`): radius from centroid p50 2.56, p99 17.4 --
   the reconstruction is unit-normalised, so view depths run ~1-20, the same
   order as the test scene's ~3. No normalisation needed. (`L` is exactly
   shift-invariant in `t`, so per-pixel re-origining is available if a future
   scene ever needs it.)

4. ~~**`brush-loss`**~~ **DONE.** Term added in `train.rs` rather than
   `brush-loss`: the distortion value is already a channel of `pred_image`, so
   it needs no kernel of its own -- `slice(4..5).mean() * weight` added to the
   loss after the LPIPS term. The `do_alpha_match` branch had to start slicing
   `0..4` explicitly, since `pred_image` is 7 channels under this pass.

5. ~~**`config.rs`**~~ **DONE.** `--distortion-weight`, default 0.0. The train
   loop selects `RasterPass::BackwardDistortion` only when it is > 0, so
   existing runs are bit-identical and pay nothing.

## Verification

**PASSED.** `crates/brush-bench-test/tests/distortion_backward.rs`, 3 tests:

* `distortion_finite_difference` -- 13 entries, all within tolerance, most
  agreeing to 3-4 significant figures. `means[*][2]` are the load-bearing
  ones: mean-z is the ONLY route by which `dL/dt` reaches a parameter, so
  they are what proves the new depth path.

      means[0][2]     numeric -3.128173e-2   analytic -3.127972e-2
      means[1][2]     numeric  9.220093e-3   analytic  9.219282e-3
      raw_opac[3][0]  numeric  1.171138e-3   analytic  1.171883e-3
      rots[1][2]      numeric -9.918585e-5   analytic -9.876459e-5

* `distortion_gradient_is_nonzero` guards against a vacuous pass -- a scene
  with no overlap produces L = 0 and every gradient 0, which finite-diff
  would "agree" with perfectly.
* `distortion_does_not_touch_color` -- `L_d` depends only on weights and
  depths, so the SH gradient must be exactly zero. Catches channel leakage.

The loss slices channel 4 ALONE. A plain `img.mean()` would hand upstream
gradient to channels 5/6 (`A_total`/`B_total`), which are saved forward state
with no gradient path, and finite-diff would disagree for a reason that is not
a bug.

## Risk

~~The backward kernel is the whole risk.~~ **Retired.** The backward is
implemented and finite-diff verified, and the default path is unchanged
(`finite_diff.rs` still passes, and the pass is only selected when
`--distortion-weight > 0`).

The remaining risk is no longer correctness but TUNING, and it is the same
risk that sank 3a: a term that provably optimises the right quantity can still
make the rendered result worse. 2DGS uses 100-1000 on normalised depth; our
scene is already unit-scale, so start ~100 and sweep. **Gate on the same
measurement that killed 3a** -- `scripts/eval_run.sh` plus LOW-altitude
bottom-third opacity and hole fraction -- not on PSNR, which stayed healthy
through 3a while the actual symptom got worse.
