#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


DATASET_NAMES = ("wlasl", "asl_citizen", "csl_large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize ISLR WebDataset mp4 samples to a fixed 224x224 resolution."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/slimelab/Projects/slt/islr/webdataset"),
        help="Root containing the original WebDataset datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/slimelab/Projects/slt/islr/webdataset_224"),
        help="Root where the resized WebDataset datasets will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to convert when they exist.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target square resize resolution.",
    )
    parser.add_argument(
        "--video-codec",
        default="libx264",
        help="ffmpeg encoder used for the resized mp4 files.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF used by libx264-style encoders.",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        help="Encoding preset passed to ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output shard.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of shard workers to run in parallel. Defaults to CPU count.",
    )
    parser.add_argument(
        "--max-shards-per-split",
        type=int,
        default=None,
        help="Optional limit for smoke tests; process only the first N shards per split.",
    )
    return parser.parse_args()


def resize_mp4_bytes(
    video_bytes: bytes,
    *,
    size: int,
    codec: str,
    crf: int,
    preset: str,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="resize_islr_wds_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.mp4"
        output_path = tmpdir_path / "output.mp4"
        input_path.write_bytes(video_bytes)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-vf",
            f"scale={size}:{size}",
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip())
        return output_path.read_bytes()


def copy_tar_member(dst_tar: tarfile.TarFile, member_name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=member_name)
    info.size = len(payload)
    dst_tar.addfile(info, io.BytesIO(payload))


def copy_json_sidecars(src_root: Path, dst_root: Path) -> None:
    for path in src_root.glob("*.json"):
        dst_path = dst_root / path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst_path)


def resize_shard(
    src_shard: Path,
    dst_shard: Path,
    *,
    size: int,
    codec: str,
    crf: int,
    preset: str,
) -> dict[str, int]:
    counts = {"videos": 0, "json": 0, "other": 0}
    with tarfile.open(src_shard, "r") as src_tar, tarfile.open(dst_shard, "w") as dst_tar:
        for member in src_tar:
            if not member.isfile():
                continue
            extracted = src_tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            if member.name.endswith(".mp4"):
                payload = resize_mp4_bytes(
                    payload,
                    size=size,
                    codec=codec,
                    crf=crf,
                    preset=preset,
                )
                counts["videos"] += 1
            elif member.name.endswith(".json"):
                counts["json"] += 1
            else:
                counts["other"] += 1
            copy_tar_member(dst_tar, member.name, payload)
    return counts


def _resize_shard_task(task: tuple[Path, Path, int, str, int, str]) -> tuple[str, dict[str, int]]:
    src_shard, dst_shard, size, codec, crf, preset = task
    counts = resize_shard(
        src_shard,
        dst_shard,
        size=size,
        codec=codec,
        crf=crf,
        preset=preset,
    )
    return dst_shard.name, counts


def convert_dataset(
    dataset_name: str,
    *,
    input_root: Path,
    output_root: Path,
    splits: tuple[str, ...],
    size: int,
    codec: str,
    crf: int,
    preset: str,
    overwrite: bool,
    num_workers: int | None,
    max_shards_per_split: int | None,
) -> dict[str, object]:
    src_root = input_root / dataset_name
    dst_root = output_root / dataset_name
    dst_root.mkdir(parents=True, exist_ok=True)
    copy_json_sidecars(src_root, dst_root)

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "source_root": str(src_root),
        "target_root": str(dst_root),
        "target_size": size,
        "splits": {},
    }

    for split in splits:
        src_split = src_root / split
        if not src_split.exists():
            continue
        dst_split = dst_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        split_counts = {"videos": 0, "json": 0, "other": 0, "shards": 0}

        shards = sorted(src_split.glob("*.tar"))
        if max_shards_per_split is not None:
            shards = shards[:max_shards_per_split]
        pending_tasks: list[tuple[Path, Path, int, str, int, str]] = []
        for src_shard in shards:
            dst_shard = dst_split / src_shard.name
            if dst_shard.exists() and not overwrite:
                split_counts["shards"] += 1
                continue
            pending_tasks.append((src_shard, dst_shard, size, codec, crf, preset))

        if pending_tasks:
            max_workers = max(1, num_workers or (os.cpu_count() or 1))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_resize_shard_task, task) for task in pending_tasks]
                for future in concurrent.futures.as_completed(futures):
                    shard_name, counts = future.result()
                    split_counts["videos"] += counts["videos"]
                    split_counts["json"] += counts["json"]
                    split_counts["other"] += counts["other"]
                    split_counts["shards"] += 1
                    print(
                        f"[{dataset_name}/{split}] finished {shard_name} "
                        f"(videos={counts['videos']}, json={counts['json']})",
                        flush=True,
                    )

        summary["splits"][split] = split_counts

    (dst_root / "resize_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for dataset_name in args.datasets:
        summary = convert_dataset(
            dataset_name,
            input_root=args.input_root,
            output_root=args.output_root,
            splits=tuple(args.splits),
            size=args.size,
            codec=args.video_codec,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
            max_shards_per_split=args.max_shards_per_split,
        )
        all_summaries.append(summary)
        print(
            f"[done] {dataset_name} -> {summary['target_root']} "
            f"(size={args.size}, splits={sorted(summary['splits'].keys())})"
        )

    (args.output_root / "resize_summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
