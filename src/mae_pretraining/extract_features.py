import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mae_pretraining.data.video_dataset import VideoDataset
from mae_pretraining.models.sign_hiera import SignHiera, load_model
import mae_pretraining.models.sign_hiera as sign_hiera


def shard_generator(data: Any, shard_size: int) -> Generator[Any, None, None]:
    for i in range(0, len(data), shard_size):
        yield data[i : i + shard_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SignHiera features from a dataset split and write a feature CSV."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset root with manifests/ and videos/")
    parser.add_argument("--output-dir", required=True, help="Directory where features and CSV are saved")
    parser.add_argument(
        "--pretrained-model-path",
        required=True,
        help="Checkpoint path for the pretrained SignHiera encoder",
    )
    parser.add_argument(
        "--model-name",
        default="hiera_base_128x224",
        help="SignHiera model name, e.g. hiera_base_128x224",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["pretrain", "finetune", "train", "val", "test"],
        help="Dataset split to extract",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Output CSV filename. Defaults to <split>_feature.csv",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=128,
        help="Clip length fed to the model. This is also the window size for sliding extraction.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Stride between extraction windows in frames. Defaults to num_frames // 2.",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Use non-overlapping windows by setting the stride equal to num_frames.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=1,
        help="Sample every n-th frame before feeding the model",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=25,
        help="Target frame rate used by the dataset sampler",
    )
    parser.add_argument(
        "--video-backend",
        default="decord",
        choices=["pyav", "video_reader", "cuda", "torchcodec", "decord"],
        help="Video backend. decord and torchcodec are faster than pyav.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference",
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 autocast on CUDA")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of videos to load per DataLoader iteration (CPU batching)",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=8,
        help="Maximum clips per GPU forward pass (controls VRAM usage)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers. Use 1 or 0 if debugging.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only extract the first N samples. Useful for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if the output feature file already exists",
    )
    parser.add_argument(
        "--save-relative-paths",
        action="store_true",
        help="Write feature_path and video_path relative to output-dir and data-dir in the CSV",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on the model for faster inference",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Use fused Triton kernel for spatial mean pooling (saves memory, allows larger batch)",
    )
    return parser.parse_args()


def resolve_window_stride(args: argparse.Namespace) -> int:
    if args.no_overlap:
        return args.num_frames
    if args.window_stride is not None:
        return args.window_stride
    return max(1, args.num_frames // 2)


def autocast_dtype(args: argparse.Namespace) -> Optional[torch.dtype]:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def load_feature_model(args: argparse.Namespace, device: torch.device) -> SignHiera:
    if args.model_name not in sign_hiera.__dict__:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    model = sign_hiera.__dict__[args.model_name](pretrained=False, strict=False)
    load_model(model, Path(args.pretrained_model_path))
    model.head = nn.Identity()
    model.eval()
    model.to(device)

    if args.compile:
        model = torch.compile(model)

    return model


def pad_collate(batch):
    """Collate function that pads variable-length clip dimensions."""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None

    max_clips = max(sample["frames"].shape[0] for sample in batch)
    frames_list = []
    padding_list = []
    num_clips_list = []
    indices = []

    for sample in batch:
        f = sample["frames"]  # [num_clips, C, T, H, W]
        p = sample["padding"]  # [num_clips]
        num_clips = f.shape[0]
        num_clips_list.append(num_clips)
        indices.append(sample["index"])

        if num_clips < max_clips:
            # Pad with zeros (these will be masked out)
            pad_f = torch.zeros(max_clips - num_clips, *f.shape[1:], dtype=f.dtype)
            f = torch.cat([f, pad_f], dim=0)
            pad_p = torch.zeros(max_clips - num_clips, dtype=p.dtype)
            p = torch.cat([p, pad_p], dim=0)

        frames_list.append(f)
        padding_list.append(p)

    return {
        "frames": torch.stack(frames_list),      # [B, max_clips, C, T, H, W]
        "padding": torch.stack(padding_list),     # [B, max_clips]
        "index": torch.tensor(indices),
        "num_clips": torch.tensor(num_clips_list),
    }


def create_dataloader(args: argparse.Namespace, stride: int) -> DataLoader:
    if args.video_backend == "cuda":
        torch.multiprocessing.set_start_method("spawn", force=True)

    dataset = VideoDataset(
        mode=args.split,
        data_dir=args.data_dir,
        video_backend=args.video_backend,
        target_fps=args.target_fps,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        rand_aug=False,
        train_random_horizontal_flip=False,
        train_random_crop=False,
        feature_extraction=True,
        feature_extraction_stride=stride,
        gpu=torch.cuda.current_device() if args.video_backend == "cuda" else None,
        max_num_samples=args.max_samples,
    )

    mp_context = "forkserver" if args.video_backend == "torchcodec" and args.num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.video_backend != "cuda",
        prefetch_factor=2 if args.num_workers > 0 else None,
        multiprocessing_context=mp_context,
        collate_fn=pad_collate,
    )


def extract_batched_features(
    model: SignHiera,
    frames: torch.Tensor,
    padding: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> List[torch.Tensor]:
    """Extract features for a batch of clips.

    Returns a list of per-clip feature tensors with padding removed.
    """
    frames = frames.to(device, non_blocking=True).float()
    padding = padding.to(device, non_blocking=True)

    dtype = autocast_dtype(args)
    use_autocast = device.type == "cuda" and dtype is not None

    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=use_autocast,
    ):
        extract_fn = model.extract_features_fused if args.fused else model.extract_features
        # extract_features removes padding per-clip and concatenates → [total_T, D]
        x = extract_fn(frames, padding=padding)

    # Compute per-clip feature sizes to split the concatenated output
    mu_t = model.mask_unit_size[0] * model.patch_stride[0]
    _, size = model.reroll.schedule[model.stage_ends[-1]]
    T_out = size[0]
    num_padding_units = padding // mu_t
    sizes = [T_out - int(n.item()) for n in num_padding_units]

    results = list(x.detach().cpu().split(sizes, dim=0))
    return results


def make_feature_path(output_dir: Path, video_path: str) -> Path:
    stem = Path(video_path).stem
    prefix = stem[:5] if stem else "misc"
    return output_dir / "features" / prefix / f"{stem}.pt"


def maybe_relpath(path: Path, start: Path, enabled: bool) -> str:
    if not enabled:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    stride = resolve_window_stride(args)

    if stride <= 0:
        raise ValueError("--window-stride must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (args.csv_name or f"{args.split}_feature.csv")

    device = torch.device(args.device)
    model = load_feature_model(args, device=device)
    dataloader = create_dataloader(args, stride=stride)
    dataset = dataloader.dataset

    rows: List[Dict[str, Any]] = []
    total_features = 0
    start_time = time.time()
    videos_processed = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        frames = batch["frames"]       # [B, max_clips, C, T, H, W]
        padding = batch["padding"]     # [B, max_clips]
        indices = batch["index"]       # [B]
        num_clips = batch["num_clips"] # [B]

        batch_size = frames.shape[0]

        # Check which videos need extraction vs skip
        video_paths = []
        feature_paths = []
        need_extract = []
        for i in range(batch_size):
            idx = int(indices[i].item())
            vp = dataset.path_to_videos[idx]
            fp = make_feature_path(output_dir, vp)
            video_paths.append(vp)
            feature_paths.append(fp)
            need_extract.append(args.overwrite or not fp.exists())

        # Extract features for videos that need it
        extracted_features = [None] * batch_size
        if any(need_extract):
            # Flatten all clips from videos that need extraction
            all_clips = []
            all_pads = []
            clip_to_video = []  # maps each clip back to its video index
            for i in range(batch_size):
                if not need_extract[i]:
                    continue
                nc = int(num_clips[i].item())
                for c in range(nc):
                    all_clips.append(frames[i, c])
                    all_pads.append(padding[i, c])
                    clip_to_video.append(i)

            all_clips = torch.stack(all_clips)  # [total_clips, C, T, H, W]
            all_pads = torch.stack(all_pads)    # [total_clips]

            # Process in sub-batches through the model
            all_feats = []
            for start in range(0, len(all_clips), args.gpu_batch_size):
                end = min(start + args.gpu_batch_size, len(all_clips))
                chunk_feats = extract_batched_features(
                    model, all_clips[start:end], all_pads[start:end], device, args
                )
                all_feats.extend(chunk_feats)

            # Group features back by video
            per_video_feats = {}
            for feat, vid_idx in zip(all_feats, clip_to_video):
                per_video_feats.setdefault(vid_idx, []).append(feat)

            for vid_idx, feat_list in per_video_feats.items():
                feat = torch.cat(feat_list, dim=0)
                extracted_features[vid_idx] = feat
                feature_paths[vid_idx].parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat, feature_paths[vid_idx])

        # Build CSV rows
        for i in range(batch_size):
            idx = int(indices[i].item())
            vp = video_paths[i]
            label = dataset.labels[idx] if idx < len(dataset.labels) else ""
            nc = int(num_clips[i].item())

            if extracted_features[i] is not None:
                features = extracted_features[i]
            else:
                features = torch.load(feature_paths[i], map_location="cpu")

            num_features = int(features.shape[0]) if features.ndim > 0 else 0
            feature_dim = int(features.shape[-1]) if features.ndim > 1 else 0
            total_features += num_features

            pad_vals = [str(int(padding[i, c].item())) for c in range(nc)]
            rows.append(
                {
                    "video_path": maybe_relpath(Path(vp), Path(args.data_dir), args.save_relative_paths),
                    "feature_path": maybe_relpath(feature_paths[i], output_dir, args.save_relative_paths),
                    "label": label,
                    "num_feature_vectors": num_features,
                    "feature_dim": feature_dim,
                    "window_size": args.num_frames,
                    "window_stride": stride,
                    "num_clips": nc,
                    "padding_frames": ",".join(pad_vals),
                }
            )

        videos_processed += batch_size

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "feature_path",
                "label",
                "num_feature_vectors",
                "feature_dim",
                "window_size",
                "window_stride",
                "num_clips",
                "padding_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start_time
    print(f"Saved {len(rows)} feature files to {output_dir / 'features'}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Total feature vectors: {total_features}")
    print(f"Elapsed time: {elapsed:.2f}s ({len(rows)/elapsed:.1f} videos/s)")


if __name__ == "__main__":
    main()
