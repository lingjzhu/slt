#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
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
        description="Compare fp32 and fp16 PEAR WDS outputs and render side-by-side validation videos."
    )
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--fp16-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--num-render-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_member_bytes(src_tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fileobj = src_tar.extractfile(member)
    if fileobj is None:
        raise FileNotFoundError(f"Failed to extract {member.name}")
    with fileobj:
        return fileobj.read()


def load_npz_map(shard: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with tarfile.open(shard, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            payload = read_member_bytes(tf, member)
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                out[member.name[:-4]] = {key: data[key] for key in data.files}
    return out


def summarize_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    diff = np.abs(a64 - b64)
    denom = np.maximum(np.abs(a64), 1e-8)
    rel = diff / denom
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean((a64 - b64) ** 2))) if diff.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "mean_rel": float(rel.mean()) if rel.size else 0.0,
    }


def aggregate_summaries(per_sample: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    metrics_by_key: dict[str, dict[str, list[float]]] = {}
    for sample in per_sample:
        for key, summary in sample.items():
            key_metrics = metrics_by_key.setdefault(key, {})
            for metric_name, value in summary.items():
                key_metrics.setdefault(metric_name, []).append(value)

    aggregated: dict[str, dict[str, float]] = {}
    for key, metrics in metrics_by_key.items():
        aggregated[key] = {}
        for metric_name, values in metrics.items():
            aggregated[key][f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[key][f"{metric_name}_max"] = float(np.max(values))
    return aggregated


def build_cameras_kwargs(batch_size: int, focal_length: float, device: torch.device) -> dict[str, torch.Tensor | float]:
    screen_size = torch.tensor([1024, 1024], device=device).float()[None].repeat(batch_size, 1)
    return {
        "principal_point": torch.zeros(batch_size, 2, device=device).float(),
        "focal_length": focal_length,
        "image_size": screen_size,
        "device": device,
    }


def reconstruct_vertices(sample: dict[str, np.ndarray], ehm_model: EHM_v2, device: torch.device) -> torch.Tensor:
    body_params = {
        key.replace("body/", ""): torch.from_numpy(value).float().to(device)
        for key, value in sample.items()
        if key.startswith("body/")
    }
    flame_params = {
        key.replace("flame/", ""): torch.from_numpy(value).float().to(device)
        for key, value in sample.items()
        if key.startswith("flame/")
    }
    with torch.inference_mode():
        return ehm_model(body_params, flame_params, pose_type="rotmat")["vertices"]


def render_pair_video(
    sample_name: str,
    fp32_sample: dict[str, np.ndarray],
    fp16_sample: dict[str, np.ndarray],
    body_renderer: BodyRenderer,
    ehm_model: EHM_v2,
    lights: PointLights,
    device: torch.device,
    out_path: Path,
) -> None:
    vertices32 = reconstruct_vertices(fp32_sample, ehm_model, device)
    vertices16 = reconstruct_vertices(fp16_sample, ehm_model, device)
    pd_cam32 = torch.from_numpy(fp32_sample["camera/pd_cam"]).float().to(device)
    pd_cam16 = torch.from_numpy(fp16_sample["camera/pd_cam"]).float().to(device)
    faces = body_renderer.faces[0]
    smplx2flame_ind = ehm_model.smplx.smplx2flame_ind
    fps = float(fp32_sample.get("target_fps", fp32_sample["source_fps"])[0])

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "faststart"],
        macro_block_size=None,
    )
    try:
        for i in range(vertices32.shape[0]):
            cam32 = GS_Camera(
                **build_cameras_kwargs(1, 24.0, device),
                R=pd_cam32[i : i + 1, :3, :3],
                T=pd_cam32[i : i + 1, :3, 3],
            )
            cam16 = GS_Camera(
                **build_cameras_kwargs(1, 24.0, device),
                R=pd_cam16[i : i + 1, :3, :3],
                T=pd_cam16[i : i + 1, :3, 3],
            )
            img32 = body_renderer.render_mesh(
                vertices32[i : i + 1],
                cam32,
                faces=faces[None],
                lights=lights,
                smplx2flame_ind=smplx2flame_ind,
            )[0, :3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            img16 = body_renderer.render_mesh(
                vertices16[i : i + 1],
                cam16,
                faces=faces[None],
                lights=lights,
                smplx2flame_ind=smplx2flame_ind,
            )[0, :3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)

            diff = np.abs(img32.astype(np.int16) - img16.astype(np.int16)).astype(np.uint8)
            panel = np.concatenate([img32, img16, diff], axis=1)
            panel[:40, :, :] = 0
            writer.append_data(panel[: panel.shape[0] - (panel.shape[0] % 2), : panel.shape[1] - (panel.shape[1] % 2)])
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_shards = sorted(path for path in args.fp32_root.rglob("*.tar"))
    fp16_shards = sorted(path for path in args.fp16_root.rglob("*.tar"))
    fp16_map = {path.relative_to(args.fp16_root).as_posix(): path for path in fp16_shards}

    per_shard_reports = []
    render_candidates: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray], float]] = []

    for fp32_shard in fp32_shards:
        rel = fp32_shard.relative_to(args.fp32_root).as_posix()
        fp16_shard = fp16_map.get(rel)
        if fp16_shard is None:
            continue

        fp32_samples = load_npz_map(fp32_shard)
        fp16_samples = load_npz_map(fp16_shard)
        common_names = sorted(set(fp32_samples) & set(fp16_samples))

        sample_reports = []
        for name in common_names:
            sample32 = fp32_samples[name]
            sample16 = fp16_samples[name]
            sample_report: dict[str, dict[str, float]] = {}
            for key in sorted(set(sample32) & set(sample16)):
                if sample32[key].dtype.kind not in {"f"}:
                    continue
                sample_report[key] = summarize_error(sample32[key], sample16[key].astype(sample32[key].dtype, copy=False))
            sample_reports.append(sample_report)

            cam_rmse = sample_report.get("camera/pd_cam", {}).get("rmse", 0.0)
            render_candidates.append((f"{rel}::{name}", sample32, sample16, cam_rmse))

        per_shard_reports.append(
            {
                "shard": rel,
                "num_common_samples": len(common_names),
                "aggregated": aggregate_summaries(sample_reports),
            }
        )

    render_candidates.sort(key=lambda item: item[3], reverse=True)
    selected = render_candidates[: min(args.num_render_samples, len(render_candidates))]

    device = torch.device(args.device)
    body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).to(device)
    ehm_model = EHM_v2("assets/FLAME", "assets/SMPLX").to(device)
    ehm_model.eval()
    lights = PointLights(device=device, location=[[0.0, -1.0, -10.0]])

    rendered = []
    for idx, (name, sample32, sample16, cam_rmse) in enumerate(selected):
        clip_id = name.replace("/", "__").replace("::", "__")
        out_path = args.render_dir / f"{idx:02d}_{clip_id}.mp4"
        render_pair_video(name, sample32, sample16, body_renderer, ehm_model, lights, device, out_path)
        rendered.append({"name": name, "video": str(out_path), "camera_rmse": cam_rmse})

    report = {
        "fp32_root": str(args.fp32_root),
        "fp16_root": str(args.fp16_root),
        "num_shards_compared": len(per_shard_reports),
        "per_shard": per_shard_reports,
        "rendered_pairs": rendered,
    }
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"num_shards_compared": len(per_shard_reports), "rendered_pairs": len(rendered)}, indent=2))


if __name__ == "__main__":
    main()
