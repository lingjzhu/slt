#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from models.modules.ehm import EHM_v2
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import ConfigDict, add_extra_cfgs
from utils.pipeline_utils import to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PEAR on WebDataset video shards and write tar-to-tar outputs "
            "containing raw features, decoded parameters, cameras, and body mesh vertices."
        )
    )
    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument(
        "--input-shard",
        type=Path,
        help="Single input shard tar to process.",
    )
    io_group.add_argument(
        "--input-root",
        type=Path,
        help="Directory containing shard tar files. Shards are discovered recursively.",
    )
    parser.add_argument(
        "--output-shard",
        type=Path,
        help="Output tar when --input-shard is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root when --input-root is used. Relative shard layout is preserved.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for PEAR inference.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=256,
        help="Maximum number of frames per model forward pass.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=32,
        help="Maximum number of clips buffered before flushing to inference.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=12.0,
        help="Target frame rate after temporal downsampling.",
    )
    parser.add_argument(
        "--shards",
        nargs="*",
        default=None,
        help=(
            "Optional shard names or relative paths when --input-root is used, "
            "for example train/00000.tar dev.tar."
        ),
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional PEAR checkpoint. Defaults to the published stage1 checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output shards.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only save latent features and model parameters (skip vertices/faces).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Save float parameters in float16 precision to cut storage size in half.",
    )
    args = parser.parse_args()

    if args.input_shard is not None and args.output_shard is None:
        parser.error("--output-shard is required with --input-shard")
    if args.input_root is not None and args.output_root is None:
        parser.error("--output-root is required with --input-root")

    return args


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


def group_members(members: list[tarfile.TarInfo]) -> dict[str, dict[str, tarfile.TarInfo]]:
    grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in members:
        if not member.isfile():
            continue
        rel_path = safe_relative_member_path(member.name)
        suffix = rel_path.suffix.lower()
        if suffix not in {".mp4", ".txt"}:
            continue
        stem = str(rel_path.with_suffix(""))
        grouped.setdefault(stem, {})[suffix] = member
    return grouped


def tar_add_bytes(dst_tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    dst_tar.addfile(info, io.BytesIO(payload))


def read_member_bytes(src_tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fileobj = src_tar.extractfile(member)
    if fileobj is None:
        raise FileNotFoundError(f"Failed to extract {member.name}")
    with fileobj:
        return fileobj.read()


def decode_video_bytes(payload: bytes) -> tuple[list[np.ndarray], float]:
    frames: list[np.ndarray] = []
    with av.open(io.BytesIO(payload), mode="r", format="mp4") as container:
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        elif stream.base_rate is not None:
            fps = float(stream.base_rate)
        else:
            fps = float(stream.guessed_rate or 24.0)
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_rgb().to_ndarray())
    if not frames:
        raise ValueError("decoded zero frames")
    return frames, fps


def select_frame_indices(total_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if total_frames <= 1 or target_fps >= source_fps:
        return np.arange(total_frames, dtype=np.int64)
    step = source_fps / target_fps
    indices = np.floor(np.arange(0, total_frames, step)).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    return np.unique(indices)


def pad_and_resize(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img
    return padded_img


def pack_npz_bytes(payload: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def member_output_name(stem: str, suffix: str) -> str:
    return f"{stem}{suffix}"


@dataclass
class PendingClip:
    stem: str
    txt_payload: bytes | None
    source_fps: float
    original_num_frames: int
    selected_indices: np.ndarray
    frame_tensors: list[torch.Tensor]


class PearMeshFeatureModel:
    def __init__(
        self,
        device: str,
        checkpoint_path: Path | None = None,
        *,
        minimal: bool = False,
        fp16: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.minimal = minimal
        self.fp16 = fp16

        meta_cfg = ConfigDict(model_config_path=str(Path("configs") / "infer.yaml"))
        meta_cfg = add_extra_cfgs(meta_cfg)
        self.model = Ehm_Pipeline(meta_cfg).to(self.device)
        self.ehm = None
        self.body_renderer = None
        self.faces = None
        if not self.minimal:
            self.ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(self.device)
            self.body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).to(self.device)

        ckpt_path = checkpoint_path
        if ckpt_path is None:
            ckpt_path = Path(
                hf_hub_download(
                    repo_id="BestWJH/PEAR_models",
                    filename="ehm_model_stage1.pt",
                    repo_type="model",
                )
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.backbone.load_state_dict(state["backbone"], strict=False)
        self.model.head.load_state_dict(state["head"], strict=False)

        self.model.eval()
        if self.ehm is not None:
            self.ehm.eval()
        if self.body_renderer is not None:
            self.faces = self.body_renderer.faces[0].detach().cpu().numpy().astype(np.int32)

    @torch.inference_mode()
    def infer_frames(self, frames: list[torch.Tensor], frame_batch_size: int) -> dict[str, np.ndarray]:
        raw_accum: dict[str, list[np.ndarray]] = {}
        body_accum: dict[str, list[np.ndarray]] = {}
        flame_accum: dict[str, list[np.ndarray]] = {}
        camera_accum: list[np.ndarray] = []
        vertices_accum: list[np.ndarray] = []
        minimal = getattr(self, "minimal", False)

        for start in range(0, len(frames), frame_batch_size):
            chunk = torch.stack(frames[start : start + frame_batch_size], dim=0).to(self.device)
            outputs = self.model(chunk)
            smplx_outputs = None
            if not minimal:
                assert self.ehm is not None
                smplx_outputs = self.ehm(outputs["body_param"], outputs["flame_param"], pose_type="aa")

            if not minimal:
                for key, value in outputs["raw_features"].items():
                    if value is None:
                        continue
                    raw_accum.setdefault(key, []).append(value.detach().cpu().numpy())
            
            for key, value in outputs["body_param"].items():
                if value is None:
                    continue
                body_accum.setdefault(key, []).append(value.detach().cpu().numpy())
            for key, value in outputs["flame_param"].items():
                if value is None:
                    continue
                flame_accum.setdefault(key, []).append(value.detach().cpu().numpy())
            camera_accum.append(outputs["pd_cam"].detach().cpu().numpy())
            if not minimal:
                assert smplx_outputs is not None
                vertices_accum.append(smplx_outputs["vertices"].detach().cpu().numpy())

        fp16 = getattr(self, "fp16", False)

        def _concat_and_cast(parts: list[np.ndarray]) -> np.ndarray:
            arr = np.concatenate(parts, axis=0)
            if fp16 and arr.dtype in [np.float32, np.float64]:
                return arr.astype(np.float16)
            return arr

        combined: dict[str, np.ndarray] = {}
        if not minimal:
            for key, parts in raw_accum.items():
                combined[f"raw/{key}"] = _concat_and_cast(parts)
        
        for key, parts in body_accum.items():
            combined[f"body/{key}"] = _concat_and_cast(parts)
        for key, parts in flame_accum.items():
            combined[f"flame/{key}"] = _concat_and_cast(parts)
        combined["camera/pd_cam"] = _concat_and_cast(camera_accum)
        if not minimal:
            assert self.faces is not None
            combined["mesh/vertices"] = _concat_and_cast(vertices_accum)
            combined["mesh/faces"] = self.faces
        return combined


def build_clip_npz_payload(
    clip: PendingClip,
    combined_outputs: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    target_fps: float,
    minimal: bool = False,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "source_fps": np.array([clip.source_fps], dtype=np.float32),
        "target_fps": np.array([target_fps], dtype=np.float32),
        "source_num_frames": np.array([clip.original_num_frames], dtype=np.int32),
        "num_frames": np.array([end_idx - start_idx], dtype=np.int32),
        "frame_indices": clip.selected_indices.astype(np.int32),
    }
    for key, value in combined_outputs.items():
        if key == "mesh/faces":
            payload[key] = value
        else:
            payload[key] = value[start_idx:end_idx]

    if minimal:
        # Remove redundant parameters that are already in body/ or flame/
        to_drop = [
            "raw/smplx_pose_6d", "raw/smplx_scale", "raw/smplx_shape",
            "raw/smplx_expression", "raw/smplx_joint_offset",
            "raw/flame_pose", "raw/flame_shape", "raw/flame_expression",
            "raw/cam_linear", "raw/cam_pred", "raw/full_proj"
        ]
        for k in to_drop:
            payload.pop(k, None)

    return payload


def flush_pending(
    pending: list[PendingClip],
    dst_tar: tarfile.TarFile,
    model: PearMeshFeatureModel,
    frame_batch_size: int,
    target_fps: float,
) -> None:
    if not pending:
        return

    flat_frames: list[torch.Tensor] = []
    clip_ranges: list[tuple[int, int]] = []
    for clip in pending:
        start = len(flat_frames)
        flat_frames.extend(clip.frame_tensors)
        end = len(flat_frames)
        clip_ranges.append((start, end))

    combined_outputs = model.infer_frames(flat_frames, frame_batch_size=frame_batch_size)

    for clip, (start, end) in zip(pending, clip_ranges):
        payload = build_clip_npz_payload(
            clip=clip,
            combined_outputs=combined_outputs,
            start_idx=start,
            end_idx=end,
            target_fps=target_fps,
            minimal=getattr(model, "minimal", False),
        )
        tar_add_bytes(dst_tar, member_output_name(clip.stem, ".npz"), pack_npz_bytes(payload))
        if clip.txt_payload is not None:
            tar_add_bytes(dst_tar, member_output_name(clip.stem, ".txt"), clip.txt_payload)

        meta = {
            "source_fps": clip.source_fps,
            "target_fps": target_fps,
            "source_num_frames": clip.original_num_frames,
            "num_frames": int(end - start),
            "frame_indices": clip.selected_indices.tolist(),
            "saved_keys": sorted(payload.keys()),
        }
        tar_add_bytes(
            dst_tar,
            member_output_name(clip.stem, ".json"),
            json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        )

    pending.clear()


def process_shard(
    src_shard: Path,
    dst_shard: Path,
    *,
    model: PearMeshFeatureModel,
    frame_batch_size: int,
    clip_batch_size: int,
    target_fps: float,
    max_samples_per_shard: int | None,
) -> dict[str, int]:
    counts = {"samples": 0, "skipped": 0}
    dst_shard.parent.mkdir(parents=True, exist_ok=True)
    # Avoid hidden temp names on FUSE/NTFS mounts; they have shown flaky finalization.
    tmp_shard = dst_shard.with_name(f"{dst_shard.name}.tmp")
    if tmp_shard.exists():
        tmp_shard.unlink()

    pending: list[PendingClip] = []
    pending_frame_count = 0

    try:
        with tarfile.open(src_shard, "r") as src_tar, tarfile.open(tmp_shard, "w") as dst_tar:
            grouped = group_members(src_tar.getmembers())

            for stem in sorted(grouped.keys()):
                if max_samples_per_shard is not None and counts["samples"] >= max_samples_per_shard:
                    break

                sample = grouped[stem]
                mp4_member = sample.get(".mp4")
                if mp4_member is None:
                    counts["skipped"] += 1
                    continue

                txt_member = sample.get(".txt")
                txt_payload = read_member_bytes(src_tar, txt_member) if txt_member is not None else None

                try:
                    frames, source_fps = decode_video_bytes(read_member_bytes(src_tar, mp4_member))
                    selected_indices = select_frame_indices(len(frames), source_fps, target_fps)
                    frame_tensors = []
                    for idx in selected_indices.tolist():
                        resized = pad_and_resize(frames[idx], target_size=256)
                        img_patch = to_tensor(resized, "cpu")
                        frame_tensors.append(torch.permute(img_patch / 255, (2, 0, 1)))
                except Exception:
                    counts["skipped"] += 1
                    continue

                pending.append(
                    PendingClip(
                        stem=stem,
                        txt_payload=txt_payload,
                        source_fps=source_fps,
                        original_num_frames=len(frames),
                        selected_indices=selected_indices,
                        frame_tensors=frame_tensors,
                    )
                )
                pending_frame_count += len(frame_tensors)
                counts["samples"] += 1

                if pending_frame_count >= frame_batch_size or len(pending) >= clip_batch_size:
                    flush_pending(
                        pending=pending,
                        dst_tar=dst_tar,
                        model=model,
                        frame_batch_size=frame_batch_size,
                        target_fps=target_fps,
                    )
                    pending_frame_count = 0

            if pending:
                flush_pending(
                    pending=pending,
                    dst_tar=dst_tar,
                    model=model,
                    frame_batch_size=frame_batch_size,
                    target_fps=target_fps,
                )

        for attempt in range(3):
            if tmp_shard.exists():
                tmp_shard.replace(dst_shard)
                break
            if dst_shard.exists():
                break
            time.sleep(1.0 * (attempt + 1))
        else:
            raise FileNotFoundError(
                f"Temporary shard disappeared before final move: {tmp_shard} -> {dst_shard}"
            )
    finally:
        if tmp_shard.exists():
            tmp_shard.unlink()

    return counts


def resolve_shards(input_root: Path, shard_names: list[str] | None) -> list[Path]:
    if shard_names:
        return [(input_root / name).resolve() for name in shard_names]
    return sorted(path.resolve() for path in input_root.rglob("*.tar"))


def process_many(args: argparse.Namespace) -> None:
    assert args.input_root is not None
    assert args.output_root is not None

    args.output_root.mkdir(parents=True, exist_ok=True)
    shards = resolve_shards(args.input_root, args.shards)
    print(f"Found {len(shards)} shard(s)")

    torch.set_float32_matmul_precision("high")
    model = PearMeshFeatureModel(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        minimal=args.minimal,
        fp16=args.fp16,
    )

    for src_shard in shards:
        rel_path = src_shard.relative_to(args.input_root.resolve())
        dst_shard = args.output_root / rel_path
        if dst_shard.exists() and not args.overwrite:
            print(f"Skipping existing {dst_shard}")
            continue

        print(f"Processing {src_shard} -> {dst_shard}")
        counts = process_shard(
            src_shard=src_shard,
            dst_shard=dst_shard,
            model=model,
            frame_batch_size=args.frame_batch_size,
            clip_batch_size=args.clip_batch_size,
            target_fps=args.target_fps,
            max_samples_per_shard=args.max_samples_per_shard,
        )
        print(f"  samples={counts['samples']} skipped={counts['skipped']}")


def process_one(args: argparse.Namespace) -> None:
    assert args.input_shard is not None
    assert args.output_shard is not None

    args.output_shard.parent.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    model = PearMeshFeatureModel(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        minimal=args.minimal,
        fp16=args.fp16,
    )

    print(f"Processing {args.input_shard} -> {args.output_shard}")
    counts = process_shard(
        src_shard=args.input_shard.resolve(),
        dst_shard=args.output_shard.resolve(),
        model=model,
        frame_batch_size=args.frame_batch_size,
        clip_batch_size=args.clip_batch_size,
        target_fps=args.target_fps,
        max_samples_per_shard=args.max_samples_per_shard,
    )
    print(f"  samples={counts['samples']} skipped={counts['skipped']}")


def main() -> None:
    args = parse_args()
    if args.input_shard is not None:
        process_one(args)
    else:
        process_many(args)


if __name__ == "__main__":
    main()
