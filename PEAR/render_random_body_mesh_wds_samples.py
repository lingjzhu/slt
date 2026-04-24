#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import imageio
import numpy as np
import torch
from pytorch3d.renderer import PointLights

from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from models.modules.ehm import EHM_v2
from utils.graphics_utils import GS_Camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select random samples from output WDS shards and render body mesh videos."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing output WDS tar files. Shards are discovered recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for rendered inspection videos.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of random WDS samples to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for rendering.",
    )
    return parser.parse_args()


def build_cameras_kwargs(batch_size: int, focal_length: float, device: torch.device) -> dict[str, torch.Tensor | float]:
    screen_size = torch.tensor([1024, 1024], device=device).float()[None].repeat(batch_size, 1)
    return {
        "principal_point": torch.zeros(batch_size, 2, device=device).float(),
        "focal_length": focal_length,
        "image_size": screen_size,
        "device": device,
    }


def safe_relative_member_path(member_name: str) -> Path:
    raw_path = Path(member_name)
    clean_parts: list[str] = []
    for part in raw_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"Refusing unsafe member path: {member_name}")
        clean_parts.append(part)
    if not clean_parts:
        raise ValueError(f"Refusing empty member path: {member_name}")
    return Path(*clean_parts)


def read_member_bytes(src_tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fileobj = src_tar.extractfile(member)
    if fileobj is None:
        raise FileNotFoundError(f"Failed to extract {member.name}")
    with fileobj:
        return fileobj.read()


def collect_npz_members(shards: list[Path]) -> list[tuple[Path, str]]:
    refs: list[tuple[Path, str]] = []
    for shard in shards:
        with tarfile.open(shard, "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                rel_path = safe_relative_member_path(member.name)
                if rel_path.suffix.lower() == ".npz":
                    refs.append((shard, str(rel_path)))
    return refs


def load_sample(shard: Path, member_name: str) -> dict[str, np.ndarray]:
    with tarfile.open(shard, "r") as tf:
        member = tf.getmember(member_name)
        payload = read_member_bytes(tf, member)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def render_video(
    sample: dict[str, np.ndarray],
    body_renderer: BodyRenderer,
    ehm_model: EHM_v2,
    lights: PointLights,
    device: torch.device,
    out_path: str,
) -> None:
    # 1. Get/Compute Vertices
    if "mesh/vertices" in sample:
        vertices = torch.from_numpy(sample["mesh/vertices"]).float().to(device)
    else:
        body_params = {
            k.replace("body/", ""): torch.from_numpy(v).float().to(device)
            for k, v in sample.items() if k.startswith("body/")
        }
        flame_params = {
            k.replace("flame/", ""): torch.from_numpy(v).float().to(device)
            for k, v in sample.items() if k.startswith("flame/")
        }
        with torch.inference_mode():
            res = ehm_model(body_params, flame_params, pose_type="rotmat")
            vertices = res["vertices"]

    # 2. Setup Camera
    if "camera/pd_cam" in sample:
        pd_cam = torch.from_numpy(sample["camera/pd_cam"]).float().to(device)
    else:
        pd_cam = torch.from_numpy(sample["raw/pd_cam"]).float().to(device)
    
    # Use actual FPS from sample if available
    fps = 24.0
    if "target_fps" in sample:
        fps = float(sample["target_fps"][0])
    elif "source_fps" in sample:
        fps = float(sample["source_fps"][0])

    # 3. Get Semantic Indices for SMPL-X rendering (Head)
    smplx2flame_ind = ehm_model.smplx.smplx2flame_ind

    # 4. Render Loop
    frames = []
    faces = sample.get("mesh/faces")
    if faces is not None:
        faces = torch.from_numpy(faces).long().to(device)
    else:
        faces = body_renderer.faces[0]

    for i in range(vertices.shape[0]):
        # Setup camera for this specific frame
        camera = GS_Camera(
            **build_cameras_kwargs(1, 24.0, device),
            R=pd_cam[i : i + 1, :3, :3],
            T=pd_cam[i : i + 1, :3, 3],
        )
        
        # Use the specialized render_mesh for part coloring
        mesh_img = body_renderer.render_mesh(
            vertices[i : i + 1],
            camera,
            faces=faces if faces.ndim == 3 else faces[None],
            lights=lights,
            smplx2flame_ind=smplx2flame_ind
        )
        # render is [1, 4, 1024, 1024], first 3 channels are RGB
        frame = (mesh_img[0, :3].permute(1, 2, 0).detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
        frames.append(frame)

    # 5. Save Video
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "faststart"],
        macro_block_size=None,
    )
    try:
        for frame in frames:
            h, w = frame.shape[:2]
            writer.append_data(frame[: h - (h % 2), : w - (w % 2)])
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(path for path in args.input_root.rglob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No tar shards found under {args.input_root}")

    refs = collect_npz_members(shards)
    if not refs:
        raise FileNotFoundError(f"No .npz samples found under {args.input_root}")

    chosen_refs = rng.sample(refs, k=min(args.num_samples, len(refs)))

    device = torch.device(args.device)
    # Initialize models
    body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).to(device)
    ehm_model = EHM_v2("assets/FLAME", "assets/SMPLX").to(device)
    ehm_model.eval()

    lights = PointLights(device=device, location=[[0.0, -1.0, -10.0]])

    manifest: list[dict[str, str | int]] = []
    for idx, (shard, member_name) in enumerate(chosen_refs):
        sample = load_sample(shard, member_name)
        clip_id = Path(member_name).with_suffix("").as_posix().replace("/", "__")
        out_path = str(args.output_dir / f"{idx:02d}_{clip_id}.mp4")
        render_video(sample, body_renderer, ehm_model, lights, device, out_path)
        manifest.append(
            {
                "index": idx,
                "shard": str(shard),
                "member": member_name,
                "video": str(out_path),
            }
        )
        print(f"[{idx + 1}/{len(chosen_refs)}] {member_name} -> {out_path}")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
