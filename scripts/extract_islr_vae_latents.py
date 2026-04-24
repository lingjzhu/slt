#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import av
import torch


DATASET_NAMES = ("wlasl", "asl_citizen", "csl_large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LeanVAE latent features from ISLR WebDataset shards without temporal "
            "downsampling and write them back as WebDataset tar files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/slimelab/Projects/slt/islr/webdataset_224"),
        help="Root containing the source mp4/json WebDataset shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/slimelab/Projects/slt/islr/webdataset_224_leanvae_latents"),
        help="Root where latent WebDataset shards will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/slimelab/Projects/Sign/LeanVAE/LeanVAE-dim16.ckpt"),
        help="LeanVAE checkpoint used for encoding.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to process when present.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for LeanVAE inference.",
    )
    parser.add_argument(
        "--chunksize-enc",
        type=int,
        default=5,
        help="Encoder chunk size passed to LeanVAE tile inference.",
    )
    parser.add_argument(
        "--chunksize-dec",
        type=int,
        default=5,
        help="Decoder chunk size passed to LeanVAE tile inference.",
    )
    parser.add_argument(
        "--max-shards-per-split",
        type=int,
        default=None,
        help="Optional limit for smoke tests; process only the first N shards per split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output shard.",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=None,
        help="Optional sample cap per shard, mainly for smoke tests.",
    )
    return parser.parse_args()


def copy_json_sidecars(src_root: Path, dst_root: Path) -> None:
    for path in src_root.glob("*.json"):
        dst_path = dst_root / path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst_path)


def decode_video_bytes(payload: bytes) -> tuple[torch.Tensor, float]:
    frames = []
    with av.open(io.BytesIO(payload), mode="r", format="mp4") as container:
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        elif stream.base_rate is not None:
            fps = float(stream.base_rate)
        else:
            fps = float(stream.guessed_rate or 25.0)
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
    if not frames:
        raise ValueError("decoded zero frames")
    return torch.stack(frames, dim=0).permute(0, 3, 1, 2).contiguous(), fps


def pad_video_to_4n_plus_1(video: torch.Tensor) -> tuple[torch.Tensor, int]:
    total_frames = int(video.shape[0])
    remainder = (total_frames - 1) % 4
    if remainder == 0:
        return video, 0

    pad_len = 4 - remainder
    first_frame = video[0].unsqueeze(0).repeat(pad_len, 1, 1, 1)
    return torch.cat([first_frame, video], dim=0), pad_len


def normalize_for_leanvae(video: torch.Tensor) -> torch.Tensor:
    # Match Sign/LeanVAE/extract_latent.py preprocessing.
    regular_size = 2.0
    return video / (127.5 * regular_size) - (1.0 / regular_size)


def tensor_to_pt_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def tar_add_bytes(dst_tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    dst_tar.addfile(info, io.BytesIO(payload))


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_caption(metadata: dict[str, Any]) -> str:
    for key in ("transcription", "gloss", "caption", "text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_leanvae(checkpoint: Path, device: str, chunksize_enc: int, chunksize_dec: int):
    try:
        import sys

        leanvae_root = "/home/slimelab/Projects/Sign/LeanVAE"
        if leanvae_root not in sys.path:
            sys.path.insert(0, leanvae_root)
        from LeanVAE import LeanVAE
    except Exception as exc:  # pragma: no cover - import path / env dependent
        raise RuntimeError(
            "Failed to import LeanVAE. Install dependencies from "
            "/home/slimelab/Projects/Sign/LeanVAE/requirements.txt."
        ) from exc

    model = LeanVAE.load_from_checkpoint(str(checkpoint), strict=False)
    model.eval().to(device).float()
    model.set_tile_inference(True)
    model.chunksize_enc = chunksize_enc
    model.chunksize_dec = chunksize_dec
    return model


def encode_latent(
    model,
    video_frames: torch.Tensor,
    *,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    padded_video, num_prepended_frames = pad_video_to_4n_plus_1(video_frames)
    model_input = normalize_for_leanvae(padded_video.float()).permute(1, 0, 2, 3).unsqueeze(0)
    with torch.inference_mode():
        latent = model.encode(model_input.to(device))
    latent = latent.squeeze(0).permute(1, 0, 2, 3).contiguous().cpu()
    latent_meta = {
        "input_num_frames": int(video_frames.shape[0]),
        "padded_num_frames": int(padded_video.shape[0]),
        "prepended_padding_frames": int(num_prepended_frames),
        "latent_num_frames": int(latent.shape[0]),
        "latent_shape": list(latent.shape),
        "dtype": str(latent.dtype).replace("torch.", ""),
    }
    return latent, latent_meta


def process_shard(
    src_shard: Path,
    dst_shard: Path,
    *,
    model,
    device: str,
    max_samples_per_shard: int | None,
) -> dict[str, int]:
    counts = {"samples": 0, "skipped": 0}
    tmp_shard = dst_shard.with_name(f".{dst_shard.name}.tmp")
    if tmp_shard.exists():
        tmp_shard.unlink()

    try:
        with tarfile.open(src_shard, "r") as src_tar, tarfile.open(tmp_shard, "w") as dst_tar:
            members = {
                member.name: member for member in src_tar.getmembers() if member.isfile()
            }
            json_members = sorted(name for name in members if name.endswith(".json"))

            for json_name in json_members:
                if max_samples_per_shard is not None and counts["samples"] >= max_samples_per_shard:
                    break
                key = Path(json_name).stem
                mp4_name = f"{key}.mp4"
                if mp4_name not in members:
                    counts["skipped"] += 1
                    continue

                json_file = src_tar.extractfile(members[json_name])
                mp4_file = src_tar.extractfile(members[mp4_name])
                if json_file is None or mp4_file is None:
                    counts["skipped"] += 1
                    continue

                metadata = json.load(json_file)
                caption = extract_caption(metadata)
                video_frames, fps = decode_video_bytes(mp4_file.read())
                latent, latent_meta = encode_latent(model, video_frames, device=device)

                metadata_out = dict(metadata)
                metadata_out["caption"] = caption
                metadata_out["video_fps"] = fps
                metadata_out["vae_latent"] = latent_meta

                tar_add_bytes(dst_tar, f"{key}.pt", tensor_to_pt_bytes(latent))
                tar_add_bytes(dst_tar, f"{key}.txt", (caption + "\n").encode("utf-8"))
                tar_add_bytes(
                    dst_tar,
                    f"{key}.json",
                    json.dumps(metadata_out, ensure_ascii=False, indent=2).encode("utf-8"),
                )
                counts["samples"] += 1
        tmp_shard.replace(dst_shard)
    except Exception:
        if tmp_shard.exists():
            tmp_shard.unlink()
        raise
    return counts


def convert_dataset(
    dataset_name: str,
    *,
    input_root: Path,
    output_root: Path,
    splits: tuple[str, ...],
    model,
    device: str,
    overwrite: bool,
    max_shards_per_split: int | None,
    max_samples_per_shard: int | None,
) -> dict[str, Any]:
    src_root = input_root / dataset_name
    dst_root = output_root / dataset_name
    dst_root.mkdir(parents=True, exist_ok=True)
    copy_json_sidecars(src_root, dst_root)

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "source_root": str(src_root),
        "target_root": str(dst_root),
        "splits": {},
    }

    for split in splits:
        src_split = src_root / split
        if not src_split.exists():
            continue

        dst_split = dst_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        shard_paths = sorted(src_split.glob("*.tar"))
        if max_shards_per_split is not None:
            shard_paths = shard_paths[:max_shards_per_split]

        split_summary = {"shards": 0, "samples": 0, "skipped": 0}
        for src_shard in shard_paths:
            dst_shard = dst_split / src_shard.name
            if dst_shard.exists() and not overwrite:
                print(f"[{dataset_name}/{split}] skipping existing {dst_shard.name}", flush=True)
                split_summary["shards"] += 1
                continue

            counts = process_shard(
                src_shard,
                dst_shard,
                model=model,
                device=device,
                max_samples_per_shard=max_samples_per_shard,
            )
            split_summary["shards"] += 1
            split_summary["samples"] += counts["samples"]
            split_summary["skipped"] += counts["skipped"]
            print(
                f"[{dataset_name}/{split}] finished {dst_shard.name} "
                f"(samples={counts['samples']}, skipped={counts['skipped']})",
                flush=True,
            )

        summary["splits"][split] = split_summary

    summary_path = dst_root / "latent_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"Requested CUDA device {args.device}, but CUDA is unavailable.")

    model = load_leanvae(
        args.checkpoint,
        args.device,
        chunksize_enc=args.chunksize_enc,
        chunksize_dec=args.chunksize_dec,
    )

    global_summary = load_json_file(args.output_root / "latent_summary.json")
    if not global_summary:
        global_summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint),
        "datasets": {},
        }
    global_summary["input_root"] = str(args.input_root)
    global_summary["output_root"] = str(args.output_root)
    global_summary["checkpoint"] = str(args.checkpoint)
    global_summary.setdefault("datasets", {})

    for dataset_name in args.datasets:
        summary = convert_dataset(
            dataset_name,
            input_root=args.input_root,
            output_root=args.output_root,
            splits=tuple(args.splits),
            model=model,
            device=args.device,
            overwrite=args.overwrite,
            max_shards_per_split=args.max_shards_per_split,
            max_samples_per_shard=args.max_samples_per_shard,
        )
        global_summary["datasets"][dataset_name] = summary

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "latent_summary.json"
    summary_path.write_text(
        json.dumps(global_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
