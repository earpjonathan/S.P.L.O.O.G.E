# fpv-splat

I flew a DJI O4 Pro FPV drone over two sites six days apart and rebuilt both
flights as one 3D Gaussian Splatting reconstruction, all on an Apple Silicon Mac
with no CUDA. There's also a WebGL viewer that flies the recorded flight paths
back through the reconstruction next to the original footage.

**Write-up with figures and a live viewer:** https://projects.jonathanearp.xyz/splat/

## Results

From COLMAP's model summary for the merged reconstruction:

| | |
|---|---|
| Source clips | 9 |
| Frames registered | 7,470 of 7,471 |
| Triangulated 3D points | 3,359,339 |
| Mean reprojection error | 0.821 px |
| Mapper wall time | 1,076 min |
| Splats in the shipped asset | 3.0 M |

The ground looked see-through whenever the camera flew low. I spent most of the
project treating that as a reconstruction problem and built four fixes for it.
The actual cause was `const znear = 0.2` in the viewer, which culled the ground
right under the camera. Dropping it to 0.01 took bottom-third holes from 51.2%
to 4.4% on the test pose, and none of the 56 poses I measured got worse. The
failed fixes are written up too, in the write-up and in `PORTFOLIO.md`.

## Pipeline

```
DJI O4 Pro video (2688x2016, 50 fps)
  -> gyro quaternions from DJI telemetry       Gyroflow CLI
  -> frame selection by rotation               scripts/select_frames2.py
  -> frame extraction at 1920x1440             scripts/extract_*.sh
  -> COLMAP SfM (CPU SIFT, custom pair lists)  scripts/run_*.sh, gen_cross_pairs2.py
  -> Brush 3DGS training on Metal              scripts/run_training.sh
  -> level, centre, cap, export .splat         scripts/ply2splat.py
  -> WebGL viewer with flight replay           viewer/
```

Frames get picked when the camera has turned 8 degrees since the last kept frame,
with a minimum gap of 2 frames and a maximum of 15. That keeps 7,471 frames out of
about 70,000. Flying straight barely changes the view and a fast yaw changes
everything, so sampling on a timer wastes frames on the straights and misses the turns.

## What's in here

| Path | What it is |
|---|---|
| `scripts/` | Every script from the project, including the one-off measurements. `raster.py` is a CPU copy of the viewer's rasteriser that I used to render and measure frames offline. `znear_audit.py` and `znear_band.py` produced the near-plane numbers above. |
| `viewer/` | Fork of [antimatter15/splat](https://github.com/antimatter15/splat) (MIT) at `ba182b5`. `ghosts.js` is new and handles flight replay with a depth prepass, and `main.js` has the camera, replay and znear changes. |
| `brush-patch/` | My changes to [Brush](https://github.com/ArthurBrussee/brush): the 2DGS depth-distortion loss, written in Rust/CubeCL from the paper, with forward and backward tests. Applies to the commit in `BASE_COMMIT`. |
| `reports/` | Small metric summaries from the SfM runs and the cross-session match checks. |
| `NOTES.md` | The full lab notebook, in order, including toolchain problems and dead ends. |
| `PORTFOLIO.md` | A shorter summary of the project with the measurements. |
| `PLAN_depth_distortion.md` | The design for the depth-distortion loss before I wrote it. |

The footage, extracted frames, COLMAP databases, trained `.ply`/`.splat` files and
camera JSON aren't in the repo. They add up to about 65 GB.

## Running it

This was built as a working folder, not a package. The scripts assume the project
sits at `~/Desktop/fpv-splat` and that the data folders (`clips/`, `gyro/`,
`frames/`, `colmap/`) are next to them.

Tools used: macOS on Apple Silicon, COLMAP 4.1.1 (CPU), ffmpeg, Gyroflow 1.6.3,
Python 3.14 with `requirements.txt`, and Brush built from source.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Brush with the depth-distortion loss
git clone https://github.com/ArthurBrussee/brush.git
git -C brush checkout "$(cat brush-patch/BASE_COMMIT)"
git -C brush apply ../brush-patch/depth-distortion.patch

# Viewer, with Range support so the footage overlay can seek
.venv/bin/python scripts/serve.py 8777
```

## Limitations

- There's no metric scale. DJI telemetry has orientation but no GPS, so distances
  are in viewer units.
- The terrain height fit in `near_diag.py` only models a single height field, so
  per-site altitude numbers aren't valid on the merged two-site scene.
- A training run takes 5 to 28 hours locally, and the GPU's 4 GiB single-buffer
  limit caps a scene at about 23.7 M splats.

## License

MIT, see `LICENSE`. Two parts come from other projects and keep their own licenses:
`viewer/` is a fork of antimatter15/splat (MIT, Copyright (c) 2023 Kevin Kwok, see
`viewer/LICENSE`), and `brush-patch/` modifies Brush, so it's under Brush's
Apache-2.0 license.
