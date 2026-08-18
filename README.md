# gaussian-pipeline

Video in, Gaussian splat out. Runs COLMAP structure-from-motion and gsplat training on a Modal A10G GPU.

## Pipeline

1. Extract frames from video with ffmpeg (8-bit RGB, configurable fps)
2. COLMAP feature extraction, sequential matching, sparse reconstruction
3. gsplat training (3D Gaussian Splatting)
4. Outputs checkpoint, validation renders, and an orbit trajectory video

## Setup

```bash
pip install modal
modal token new
```

## Usage

```bash
modal run train.py --input data/scene.mov --scene-name myscene --fps 4 --iterations 7000
```

Arguments:
- `--input` path to a video file
- `--scene-name` name for the output directory
- `--fps` frame extraction rate (4 is a reasonable default)
- `--iterations` training steps (7000 quick, 30000 standard)

## Getting results

Outputs land in the `gaussian-outputs` Outputs land in the `h
mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmt gmmmmmmmmmmmmmmmmmmmmmme/mmmmmmmmenders/val_step6999_0000.png ./val.png
```

Note the single-file form. Pulling a whole directory needs a trailing slash on the destination or Modal errors with `IsADirectoryError`.

## Results

77-image chair capture, 7000 iterations:

| Metric | Value |
|---|---|
| PSNR | 24.07 |
| SSIM | 0.7343 |
| LPIPS | 0.251 |
| Gaussians | 1,608,265 |
| Training time | ~11 min on A10G |

## Gotchas

**gsplat is pinned to 1.5.3.** Building from HEAD fails against torch 2.4 (`c10d::wait_tensor` missing). The wheel avoids compiling from source entirely.

**fused-ssim is patched out.** The container reports CUDA 12.4 while torch was built for 13.0, so the extension won't compile. `simple_trainer.py` is patched at runtime to use pure-torch SSIM instead. Slower, but works.

**Input frames must be uniform dimensions.** Mixed image sizes give COLMAP inconsistent intrinsics and degrade reconstruction. Normalize with ffmpeg scale/pad before building the input video.

**10-bit HDR video needs tonemapping.** ffmpeg writes 16-bit PNGs from 10-bit HEVC, which COLMAP cannot read. The extraction step forces 8-bit RGB output.

**Capture coverage determines quality.** A tight orbit around one object leaves the background unconstrained, and those regions render as grey fog regardless of training length. Wider arcs give the background the parallax it needs.
