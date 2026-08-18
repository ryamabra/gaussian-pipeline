import modal
import os
import shutil
import subprocess
from pathlib import Path

app = modal.App("gaussian-splat-pipeline")

GSPLAT_TAG = "v1.5.3"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .apt_install(
        "ffmpeg",
        "colmap",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
    )
    .pip_install(
        f"gsplat=={GSPLAT_TAG.lstrip('v')}",
        # Old pycolmap with SceneManager
        "git+https://github.com/rmbrualla/pycolmap.git@cc7ea4b7301720ac29287dbe450952511b32125e",
        "imageio[ffmpeg]",
        "numpy<2",
        "opencv-python-headless",
        "pillow",
        "plyfile",
        "scikit-learn",
        "tqdm",
        "tyro",
        "viser",
        "torchvision",
        "tensorboard",
        "torchmetrics",
        "lpips",
        "matplotlib",
        "piexif",
        "pyyaml",
        "tensorly",
        "nerfview",
        "splines",
        # no fused-ssim — we patch the trainer to use a torch fallback
    )
    .env({
        "QT_QPA_PLATFORM": "offscreen",
        "DISPLAY": "",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    })
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=True)


# Pure-torch SSIM used as drop-in replacement for fused_ssim
TORCH_SSIM_PATCH = r'''
# --- patched: pure torch SSIM (no fused_ssim needed) ---
import torch
import torch.nn.functional as F

def _gaussian_window(window_size, sigma, channel, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def fused_ssim(img1, img2, padding="same", train=True):
    """Drop-in replacement for fused_ssim.fused_ssim"""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    # expect [B, C, H, W] or [B, H, W, C]
    if img1.shape[-1] in (1, 3) and img1.shape[1] not in (1, 3):
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)
    channel = img1.shape[1]
    window_size = 11
    sigma = 1.5
    window = _gaussian_window(window_size, sigma, channel, img1.device, img1.dtype)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
'''


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={"/outputs": volume},
)
def train_splats(
    video_bytes: bytes,
    scene_name: str = "scene",
    iterations: int = 7000,
    fps: float = 4.0,
):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["DISPLAY"] = ""

    work_dir = Path("/tmp/work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    images_dir = work_dir / "images"
    images_dir.mkdir()

    video_path = work_dir / "input.mov"
    video_path.write_bytes(video_bytes)

    # ---------- 1. Extract frames ----------
    print(f"Extracting frames at {fps} fps as 8-bit RGB...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps},format=rgb24",
            "-pix_fmt", "rgb24",
            str(images_dir / "frame_%05d.png"),
        ],
        check=True,
        capture_output=True,
    )

    num_images = len(list(images_dir.glob("*.png")))
    print(f"Extracted {num_images} frames")
    if num_images < 10:
        raise RuntimeError(f"Only {num_images} frames. Try higher --fps or a longer video.")

    from PIL import Image
    sample = Image.open(next(images_dir.glob("*.png")))
    print(f"Sample image: mode={sample.mode}, size={sample.size}")

    # ---------- 2. COLMAP ----------
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir()

    print("Running COLMAP feature_extractor...")
    subprocess.run(
        [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.max_image_size", "1600",
        ],
        check=True,
    )

    print("Running COLMAP sequential_matcher...")
    subprocess.run(
        [
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "0",
            "--SequentialMatching.overlap", "15",
            "--SequentialMatching.quadratic_overlap", "1",
        ],
        check=True,
    )

    print("Running COLMAP mapper...")
    result = subprocess.run(
        [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_num_matches", "10",
            "--Mapper.init_min_num_inliers", "30",
            "--Mapper.init_min_tri_angle", "1.0",
            "--Mapper.abs_pose_min_num_inliers", "15",
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",
            "--Mapper.filter_max_reproj_error", "6.0",
            "--Mapper.filter_min_tri_angle", "0.5",
        ],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print(result.stderr[-1500:] if len(result.stderr) > 1500 else result.stderr)

    model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        raise RuntimeError(
            "COLMAP mapper produced no models. "
            "Video likely has too little camera movement."
        )

    print(f"COLMAP finished. Models: {[d.name for d in model_dirs]}")

    def model_size(d):
        pts = d / "points3D.bin"
        return pts.stat().st_size if pts.exists() else 0

    best_model = max(model_dirs, key=model_size)
    final_sparse = sparse_dir / "0"
    if best_model != final_sparse:
        if final_sparse.exists():
            shutil.rmtree(final_sparse)
        shutil.move(str(best_model), str(final_sparse))

    # Save COLMAP to volume
    colmap_out = Path("/outputs") / scene_name / "colmap"
    if colmap_out.exists():
        shutil.rmtree(colmap_out)
    colmap_out.mkdir(parents=True)
    shutil.copytree(images_dir, colmap_out / "images")
    shutil.copytree(final_sparse, colmap_out / "sparse" / "0")
    print(f"COLMAP data saved to /outputs/{scene_name}/colmap")

    # ---------- 3. gsplat training ----------
    print(f"Cloning gsplat examples at tag {GSPLAT_TAG}...")
    gsplat_dir = Path("/tmp/gsplat")
    if gsplat_dir.exists():
        shutil.rmtree(gsplat_dir)

    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--branch", GSPLAT_TAG,
            "https://github.com/nerfstudio-project/gsplat.git",
            str(gsplat_dir),
        ],
        check=True,
        capture_output=True,
    )

    examples_dir = gsplat_dir / "examples"

    # Patch simple_trainer.py: replace fused_ssim import with pure-torch version
    trainer_path = examples_dir / "simple_trainer.py"
    src = trainer_path.read_text()
    # Remove the fused_ssim import line(s)
    src = src.replace("from fused_ssim import fused_ssim\n", "")
    src = src.replace("from fused_ssim import fused_ssim", "")
    # Inject our pure-torch implementation near the top (after imports)
    inject_marker = "import torch\n"
    if inject_marker in src:
        src = src.replace(inject_marker, inject_marker + TORCH_SSIM_PATCH + "\n", 1)
    else:
        src = TORCH_SSIM_PATCH + "\n" + src
    trainer_path.write_text(src)
    print("Patched simple_trainer.py to use pure-torch SSIM (no fused-ssim)")

    result_dir = Path("/outputs") / scene_name / "gsplat"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting gsplat training for {iterations} steps...")
    cmd = [
        "python", "simple_trainer.py", "default",
        "--data_dir", str(work_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(iterations),
        "--save_steps", str(iterations),
        "--eval_steps", str(iterations),
        "--disable_viewer",
        "--data_factor", "1",
    ]
    print("Command:", " ".join(cmd))

    proc = subprocess.run(cmd, cwd=str(examples_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"gsplat training failed with code {proc.returncode}")

    print("Training finished.")
    print("Contents of result dir:")
    for p in sorted(result_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(result_dir)}  ({p.stat().st_size} bytes)")

    return {
        "scene": scene_name,
        "num_images": num_images,
        "status": "trained",
        "colmap_path": f"/outputs/{scene_name}/colmap",
        "gsplat_path": f"/outputs/{scene_name}/gsplat",
        "iterations": iterations,
    }


@app.local_entrypoint()
def main(
    input: str = "data/IMG_5110.mov",
    scene_name: str = "test",
    iterations: int = 7000,
    fps: float = 4.0,
):
    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"No such file or directory: {input}")

    print(f"Uploading {input_path} ...")
    video_bytes = input_path.read_bytes()

    result = train_splats.remote(
        video_bytes,
        scene_name=scene_name,
        iterations=iterations,
        fps=fps,
    )
    print("Done:", result)
    print("\nDownload with:")
    print(f"  modal volume get gaussian-outputs {scene_name} ./splat_{scene_name}")
