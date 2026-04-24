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
from pytorch3d.renderer import PointLights, look_at_view_transform

from models.modules.ehm import EHM_v2
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from render_random_body_mesh_wds_samples import (
    build_cameras_kwargs,
    collect_npz_members,
    load_sample,
)
from utils.graphics_utils import GS_Camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed-view and orbit SMPL-X videos from extracted WDS samples."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["asl_citizen", "csl_large", "wlasl"],
        help="Datasets to sample from under the extracted shard root.",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=["front", "left", "right", "back"],
        help="Fixed views to render.",
    )
    parser.add_argument("--orbit-frames", type=int, default=96)
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--focal-length", type=float, default=10.0)
    parser.add_argument("--distance-scale", type=float, default=10.0)
    parser.add_argument("--min-distance", type=float, default=10.0)
    return parser.parse_args()


def _read_member_bytes(src_tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fileobj = src_tar.extractfile(member)
    if fileobj is None:
        raise FileNotFoundError(f"Failed to extract {member.name}")
    with fileobj:
        return fileobj.read()


def _load_txt_payload(shard: Path, stem: str) -> bytes | None:
    txt_name = f"{stem}.txt"
    with tarfile.open(shard, "r") as tf:
        try:
            member = tf.getmember(txt_name)
        except KeyError:
            return None
        return _read_member_bytes(tf, member)


def reconstruct_vertices(
    sample: dict[str, np.ndarray], ehm_model: EHM_v2, device: torch.device
) -> torch.Tensor:
    if "mesh/vertices" in sample:
        return torch.from_numpy(sample["mesh/vertices"]).float().to(device)

    body_params = {
        k.replace("body/", ""): torch.from_numpy(v).float().to(device)
        for k, v in sample.items()
        if k.startswith("body/")
    }
    flame_params = {
        k.replace("flame/", ""): torch.from_numpy(v).float().to(device)
        for k, v in sample.items()
        if k.startswith("flame/")
    }
    with torch.inference_mode():
        return ehm_model(body_params, flame_params, pose_type="rotmat")["vertices"]


def canonicalize_vertices(vertices: torch.Tensor) -> tuple[torch.Tensor, float]:
    mins = vertices.amin(dim=(0, 1))
    maxs = vertices.amax(dim=(0, 1))
    center = (mins + maxs) / 2.0
    centered = vertices - center.view(1, 1, 3)
    radius = centered.norm(dim=-1).amax().item()
    return centered, radius


def build_view_camera(
    batch_size: int,
    azim: float,
    elev: float,
    dist: float,
    device: torch.device,
    focal_length: float,
) -> GS_Camera:
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    R = R.expand(batch_size, -1, -1).contiguous()
    T = T.expand(batch_size, -1).contiguous()
    return GS_Camera(
        **build_cameras_kwargs(batch_size, focal_length, device),
        R=R,
        T=T,
    )


def render_frames(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    azims: list[float],
    elev: float,
    dist: float,
    focal_length: float,
    body_renderer: BodyRenderer,
    lights: PointLights,
    smplx2flame_ind: torch.Tensor,
) -> list[np.ndarray]:
    device = vertices.device
    frames: list[np.ndarray] = []
    for frame_idx, azim in enumerate(azims):
        camera = build_view_camera(
            batch_size=1,
            azim=azim,
            elev=elev,
            dist=dist,
            device=device,
            focal_length=focal_length,
        )
        mesh_img = body_renderer.render_mesh(
            vertices[frame_idx : frame_idx + 1],
            camera,
            faces=faces[None],
            lights=lights,
            smplx2flame_ind=smplx2flame_ind,
        )
        frame = (
            mesh_img[0, :3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
        )
        frame = np.ascontiguousarray(np.flipud(frame))
        frames.append(frame)
    return frames


def write_video(frames: list[np.ndarray], out_path: Path, fps: float) -> None:
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


def choose_samples(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    rng = random.Random(args.seed)
    chosen: list[tuple[str, Path, str]] = []
    per_dataset = max(1, args.num_samples // max(1, len(args.datasets)))
    remainder = args.num_samples
    for dataset in args.datasets:
        shards = sorted((args.input_root / dataset).rglob("*.tar"))
        if not shards:
            continue
        refs = collect_npz_members(shards)
        if not refs:
            continue
        pick_count = min(per_dataset, len(refs), remainder)
        for shard, member_name in rng.sample(refs, k=pick_count):
            chosen.append((dataset, shard, member_name))
        remainder -= pick_count
    if remainder > 0:
        all_refs: list[tuple[str, Path, str]] = []
        for dataset in args.datasets:
            shards = sorted((args.input_root / dataset).rglob("*.tar"))
            refs = collect_npz_members(shards)
            all_refs.extend((dataset, shard, member_name) for shard, member_name in refs)
        seen = {(dataset, str(shard), member) for dataset, shard, member in chosen}
        leftovers = [
            item
            for item in all_refs
            if (item[0], str(item[1]), item[2]) not in seen
        ]
        if leftovers:
            chosen.extend(rng.sample(leftovers, k=min(remainder, len(leftovers))))
    return chosen[: args.num_samples]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_samples(args)
    if not selected:
        raise FileNotFoundError("No WDS samples found to render.")

    device = torch.device(args.device)
    body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).to(device)
    ehm_model = EHM_v2("assets/FLAME", "assets/SMPLX").to(device)
    ehm_model.eval()
    lights = PointLights(device=device, location=[[0.0, -1.0, -10.0]])
    smplx2flame_ind = ehm_model.smplx.smplx2flame_ind

    view_azims = {
        "front": 180.0,
        "left": 90.0,
        "right": -90.0,
        "back": 0.0,
    }

    manifest: list[dict[str, object]] = []
    for idx, (dataset, shard, member_name) in enumerate(selected):
        sample = load_sample(shard, member_name)
        vertices = reconstruct_vertices(sample, ehm_model, device)
        vertices, radius = canonicalize_vertices(vertices)
        dist = max(args.min_distance, radius * args.distance_scale)

        faces = sample.get("mesh/faces")
        if faces is not None:
            faces_t = torch.from_numpy(faces).long().to(device)
        else:
            faces_t = body_renderer.faces[0].to(device)

        fps = float(sample.get("target_fps", sample.get("source_fps", np.array([24.0])))[0])
        stem = Path(member_name).with_suffix("").name
        sample_dir = args.output_dir / f"{idx:02d}_{dataset}_{stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        txt_payload = _load_txt_payload(shard, Path(member_name).with_suffix("").as_posix())
        if txt_payload is not None:
            (sample_dir / "label.txt").write_bytes(txt_payload)

        rendered_files: dict[str, str] = {}
        fixed_azims = list(range(vertices.shape[0]))
        for view_name in args.views:
            azim = view_azims[view_name]
            frames = render_frames(
                vertices,
                faces_t,
                azims=[azim] * vertices.shape[0],
                elev=args.elev,
                dist=dist,
                focal_length=args.focal_length,
                body_renderer=body_renderer,
                lights=lights,
                smplx2flame_ind=smplx2flame_ind,
            )
            out_path = sample_dir / f"{view_name}.mp4"
            write_video(frames, out_path, fps)
            rendered_files[view_name] = str(out_path)

        orbit_azims = np.linspace(180.0, 540.0, num=vertices.shape[0], endpoint=False).tolist()
        orbit_frames = render_frames(
            vertices,
            faces_t,
            azims=orbit_azims,
            elev=args.elev,
            dist=dist,
            focal_length=args.focal_length,
            body_renderer=body_renderer,
            lights=lights,
            smplx2flame_ind=smplx2flame_ind,
        )
        orbit_path = sample_dir / "orbit.mp4"
        write_video(orbit_frames, orbit_path, fps)
        rendered_files["orbit"] = str(orbit_path)

        item = {
            "index": idx,
            "dataset": dataset,
            "shard": str(shard),
            "member": member_name,
            "output_dir": str(sample_dir),
            "videos": rendered_files,
        }
        manifest.append(item)
        print(json.dumps(item))

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
