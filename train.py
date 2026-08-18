import modal

app = modal.App("gaussian-splatting")

volume = modal.Volume.from_name("splat-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "cmake",
        "ninja-build",
        "build-essential",
        "colmap",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "pillow",
        "opencv-python-headless",
        "plyfile",
        "tqdm",
        "imageio[ffmpeg]",
        "scikit-learn",
        "tyro",
        "viser",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git /opt/gsplat",
        "pip install --no-build-isolation tyro Pillow tensorboard tqdm torchmetrics"
        " opencv-python pyyaml matplotlib scikit-learn pycolmap==3.10.0"
        " imageio[ffmpeg] viser nerfview",
        "pip install wheel ninja",
        "pip install --no-build-isolation /opt/gsplat",
    )
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/outputs": volume},
)
def train_splats(files: dict[str, bytes], scene_name: str = "scene", iterations: int = 7000) -> bytes:
    import os
    import subprocess
    from pathlib import Path

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    work_dir = Path("/tmp/work")
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for name, data in files.items():
        ext = Path(name).suffix.lower()
        if ext in VIDEO_EXTS:
            video_path = work_dir / name
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(data)
            stem = Path(name).stem
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vf",
                    "fps=4,zscale=t=linear:npl=100,format=gbrpf32le,"
                    "zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
                    "zscale=t=bt709:m=bt709:r=tv,format=rgb24",
                    "-pix_fmt",
                    "rgb24",
                    str(images_dir / f"{stem}_%05d.png"),
                ],
                check=True,
            )
        else:
            (images_dir / Path(name).name).write_bytes(data)

    num_images = len(list(images_dir.glob("*")))
    if num_images < 20:
        raise ValueError(
            f"Need at least 20 images for reliable reconstruction, got {num_images}. "
            "Provide a longer video or more photos with wider coverage of the scene."
        )

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            "8",
        ],
        check=True,
    )
    subprocess.run(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.num_threads",
            "8",
        ],
        check=True,
    )
    subprocess.run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        check=True,
    )

    result_dir = work_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "python",
            "/opt/gsplat/examples/simple_trainer.py",
            "default",
            "--data-dir",
            str(work_dir),
            "--result-dir",
            str(result_dir),
            "--max-steps",
            str(iterations),
            "--save-ply",
        ],
        check=True,
        cwd="/opt/gsplat/examples",
    )

    ply_files = sorted(result_dir.rglob("*.ply"))
    if not ply_files:
        raise FileNotFoundError("No .ply file produced by trainer")
    final_ply = ply_files[-1]

    out_path = Path("/outputs") / f"{scene_name}.ply"
    out_path.write_bytes(final_ply.read_bytes())
    volume.commit()

    return final_ply.read_bytes()


@app.local_entrypoint()
def main(input: str = "data", scene_name: str = "scene", iterations: int = 7000):
    from pathlib import Path

    input_path = Path(input)
    files: dict[str, bytes] = {}

    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.is_file():
                files[p.name] = p.read_bytes()
    elif input_path.is_file():
        files[input_path.name] = input_path.read_bytes()
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

    result_bytes = train_splats.remote(files, scene_name=scene_name, iterations=iterations)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_name}.ply"
    out_path.write_bytes(result_bytes)
    print(out_path)
