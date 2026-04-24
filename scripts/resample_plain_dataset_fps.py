#!/usr/bin/env python3
"""
Resample plain sign-language video datasets to a new FPS while preserving
spatial resolution and rebuilding paired manifests.

For each input split directory:
- creates a sibling output directory with a configurable suffix
- mirrors the `videos/...` layout
- rewrites `manifests/paired_manifest.part-XXX.tsv`
- rewrites `manifests/paired_manifest.tsv`

Example:
  python slt/scripts/resample_plain_dataset_fps.py \
    --split-root /mnt/data2/sign_language_24fps/processed_24fps/all_train_plain_v3 \
    --split-root /mnt/data2/sign_language_24fps/processed_24fps/all_val_plain_v3 \
    --split-root /mnt/data2/sign_language_24fps/processed_24fps/all_test_plain_v3 \
    --target-fps 13 \
    --workers 30
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        action="append",
        required=True,
        help="Input split root, e.g. /path/to/all_train_plain_v3",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        required=True,
        help="Target output fps.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=30,
        help="Number of worker processes / ffmpeg jobs to run in parallel.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_13fps",
        help="Suffix appended to each split directory name for outputs.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="veryfast",
        help="libx264 preset for transcoding.",
    )
    parser.add_argument(
        "--ffmpeg-crf",
        type=int,
        default=18,
        help="libx264 CRF value.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing output videos when present.",
    )
    parser.add_argument(
        "--overwrite-manifests",
        action="store_true",
        help="Regenerate manifests even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debugging limit per split.",
    )
    return parser.parse_args()


def split_output_root(split_root: Path, suffix: str) -> Path:
    return split_root.parent / f"{split_root.name}{suffix}"


def run_ffmpeg(
    src_video: Path,
    dst_video: Path,
    target_fps: float,
    preset: str,
    crf: int,
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = dst_video.with_name(dst_video.stem + ".tmp" + dst_video.suffix)
    if tmp_video.exists():
        tmp_video.unlink()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-threads",
        "1",
        "-i",
        str(src_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"fps={target_fps}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        str(tmp_video),
    ]
    subprocess.run(cmd, check=True)
    tmp_video.replace(dst_video)


def chunk_manifest_lines(manifest_path: Path, chunk_paths: list[Path], limit: int | None) -> int:
    for path in chunk_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    handles = [path.open("w", encoding="utf-8", newline="") for path in chunk_paths]
    try:
        with manifest_path.open("r", encoding="utf-8") as src:
            for idx, line in enumerate(src):
                if limit is not None and idx >= limit:
                    break
                handles[idx % len(handles)].write(line)
                total += 1
    finally:
        for handle in handles:
            handle.close()
    return total


def remap_video_path(src_split_root: Path, dst_split_root: Path, raw_path: str) -> tuple[Path, str]:
    src_path = Path(raw_path)
    try:
        rel = src_path.resolve().relative_to(src_split_root.resolve())
    except Exception as exc:
        raise ValueError(f"Video path {src_path} is not under split root {src_split_root}") from exc
    dst_path = dst_split_root / rel
    return dst_path, str(dst_path.resolve())


def process_manifest_chunk(
    chunk_path: str,
    output_part_path: str,
    src_split_root: str,
    dst_split_root: str,
    target_fps: float,
    preset: str,
    crf: int,
    skip_existing: bool,
) -> dict[str, int | str]:
    chunk = Path(chunk_path)
    output_part = Path(output_part_path)
    src_root = Path(src_split_root)
    dst_root = Path(dst_split_root)

    output_part.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    reused = 0
    failed = 0

    with chunk.open("r", encoding="utf-8") as src, output_part.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 4)
            if len(parts) < 5:
                failed += 1
                continue

            raw_video_path, duration, dataset_name, language, text = parts
            src_video = Path(raw_video_path)

            try:
                dst_video, dst_video_str = remap_video_path(src_root, dst_root, raw_video_path)
                if skip_existing and dst_video.exists() and dst_video.stat().st_size > 0:
                    reused += 1
                else:
                    run_ffmpeg(
                        src_video=src_video,
                        dst_video=dst_video,
                        target_fps=target_fps,
                        preset=preset,
                        crf=crf,
                    )

                dst.write(
                    "\t".join([dst_video_str, duration, dataset_name, language, text]) + "\n"
                )
                processed += 1
            except Exception as exc:
                failed += 1
                print(
                    f"[chunk {chunk.name}] failed line={line_no} video={src_video}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "chunk": chunk.name,
        "processed": processed,
        "reused": reused,
        "failed": failed,
        "output_part": str(output_part),
    }


def merge_parts(part_paths: list[Path], merged_path: Path) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8", newline="") as out:
        for part_path in sorted(part_paths):
            if not part_path.exists():
                continue
            with part_path.open("r", encoding="utf-8") as part:
                shutil.copyfileobj(part, out)


def process_split(
    split_root: Path,
    target_fps: float,
    workers: int,
    output_suffix: str,
    preset: str,
    crf: int,
    skip_existing: bool,
    overwrite_manifests: bool,
    limit: int | None,
) -> None:
    split_root = split_root.resolve()
    src_manifest = split_root / "manifests" / "paired_manifest.tsv"
    if not src_manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {src_manifest}")

    dst_root = split_output_root(split_root, output_suffix).resolve()
    dst_manifests = dst_root / "manifests"
    staging_dir = dst_manifests / ".staging"
    merged_manifest = dst_manifests / "paired_manifest.tsv"

    dst_manifests.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, workers)
    chunk_paths = [staging_dir / f"input.part-{idx:03d}.tsv" for idx in range(num_workers)]
    part_paths = [dst_manifests / f"paired_manifest.part-{idx:03d}.tsv" for idx in range(num_workers)]

    if overwrite_manifests or not merged_manifest.exists():
        total_lines = chunk_manifest_lines(src_manifest, chunk_paths, limit)
    else:
        total_lines = 0

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_path, part_path in zip(chunk_paths, part_paths):
            if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                continue
            futures.append(
                executor.submit(
                    process_manifest_chunk,
                    chunk_path=str(chunk_path),
                    output_part_path=str(part_path),
                    src_split_root=str(split_root),
                    dst_split_root=str(dst_root),
                    target_fps=target_fps,
                    preset=preset,
                    crf=crf,
                    skip_existing=skip_existing,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())

    merge_parts([Path(result["output_part"]) for result in results], merged_manifest)

    processed = sum(int(result["processed"]) for result in results)
    reused = sum(int(result["reused"]) for result in results)
    failed = sum(int(result["failed"]) for result in results)
    print(
        f"[{split_root.name}] done total_lines={total_lines or 'existing_chunks'} "
        f"processed={processed} reused={reused} failed={failed} output={dst_root}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    for split_root in args.split_root:
        process_split(
            split_root=split_root,
            target_fps=args.target_fps,
            workers=args.workers,
            output_suffix=args.output_suffix,
            preset=args.ffmpeg_preset,
            crf=args.ffmpeg_crf,
            skip_existing=args.skip_existing,
            overwrite_manifests=args.overwrite_manifests,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
