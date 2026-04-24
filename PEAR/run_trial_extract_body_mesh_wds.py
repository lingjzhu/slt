#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from extract_csl_daily_wds_features import PearMeshFeatureModel, process_shard


DATASETS = [
    "YoutubeASL_processed",
    "how2sign_24fps",
    "bobsl_24fps",
    "csl_news",
    "openasl_24fps",
    "bobsl_24fps_manual",
    "dailymoth-70h",
    "csl_daily_24fps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small randomized PEAR body-mesh extraction trial on several datasets."
    )
    parser.add_argument(
        "--input-base",
        type=Path,
        default=Path("/mnt/data2/sign_language_24fps/processed_24fps"),
        help="Base directory containing the dataset folders.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/mnt/data2/sign_language_24fps/processed_24fps_bodymesh_trial"),
        help="Base output directory for trial shards.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=12,
        help="How many samples to extract from each chosen shard.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=12.0,
        help="Target frame rate after temporal downsampling.",
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
        "--device",
        default="cuda:0",
        help="Torch device for PEAR inference.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional PEAR checkpoint. Defaults to the published stage1 checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used to choose one shard per dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trial shards.",
    )
    return parser.parse_args()


def choose_random_shard(dataset_root: Path, rng: random.Random) -> Path:
    shards = sorted(path for path in dataset_root.rglob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No tar shards found under {dataset_root}")
    return rng.choice(shards)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_base.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    model = PearMeshFeatureModel(device=args.device, checkpoint_path=args.checkpoint_path)

    for dataset_name in DATASETS:
        dataset_root = args.input_base / dataset_name
        chosen_shard = choose_random_shard(dataset_root, rng)
        rel_path = chosen_shard.relative_to(dataset_root)
        dst_shard = args.output_base / dataset_name / rel_path

        if dst_shard.exists() and not args.overwrite:
            print(f"Skipping existing {dst_shard}")
            continue

        print(f"[{dataset_name}] {chosen_shard} -> {dst_shard}")
        counts = process_shard(
            src_shard=chosen_shard,
            dst_shard=dst_shard,
            model=model,
            frame_batch_size=args.frame_batch_size,
            clip_batch_size=args.clip_batch_size,
            target_fps=args.target_fps,
            max_samples_per_shard=args.samples_per_shard,
        )
        print(f"  samples={counts['samples']} skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
