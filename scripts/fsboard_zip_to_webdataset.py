#!/usr/bin/env python3
"""Convert fsboard.zip clips into WebDataset tar shards.

The script never expands the full archive on disk. Each worker process opens the
zip file on demand, transcodes one clip at a time with ffmpeg, and writes
WebDataset samples directly into tar shards.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile


DEFAULT_VIDEO_FILTER = (
    "fps=24,"
    "scale=224:224:force_original_aspect_ratio=decrease,"
    "pad=224:224:(ow-iw)/2:(oh-ih)/2:black"
)


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    key: str
    zip_member: str
    transcript: str
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", default="/mnt/data2/fsboard.zip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--max-files-per-shard",
        type=int,
        default=10_000,
        help="Hard cap on files inside each tar shard.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of top-level datasets to export, e.g. daun_v3 dmk_v3.",
    )
    parser.add_argument(
        "--write-json-metadata",
        action="store_true",
        help="Store an extra .json sidecar per sample with the original metadata.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to each split.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--video-filter",
        default=DEFAULT_VIDEO_FILTER,
        help="ffmpeg -vf filter graph. Default preserves aspect ratio with padding.",
    )
    return parser.parse_args()


def chunked(items: list[Sample], size: int) -> Iterable[list[Sample]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def build_samples(
    zip_path: str,
    datasets_filter: set[str] | None,
    include_json_metadata: bool,
    max_samples_per_split: int | None,
) -> tuple[dict[str, dict[str, list[Sample]]], int]:
    with ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        samples_by_dataset: dict[str, dict[str, list[Sample]]] = {}
        files_per_sample = 3 if include_json_metadata else 2
        metadata_members = sorted(
            member
            for member in members
            if member.count("/") == 2
            and member.endswith(".json")
            and "/metadata/" in member
        )

        for metadata_member in metadata_members:
            dataset, _, filename = metadata_member.split("/")
            if datasets_filter is not None and dataset not in datasets_filter:
                continue
            split_suffix = filename.removeprefix(f"{dataset}-").removesuffix(".json")
            records = json.loads(archive.read(metadata_member))
            samples: list[Sample] = []
            video_prefix = f"{dataset}/video_clips/{dataset}-{split_suffix}/"
            for record in records:
                clip_filename = record["clipFilename"]
                zip_member = f"{video_prefix}{clip_filename}"
                if zip_member not in members:
                    raise FileNotFoundError(
                        f"Missing clip {zip_member!r} referenced by {metadata_member!r}"
                    )
                samples.append(
                    Sample(
                        dataset=dataset,
                        split=split_suffix,
                        key=Path(clip_filename).stem,
                        zip_member=zip_member,
                        transcript=str(record.get("phrase", "")).strip(),
                        metadata=record,
                    )
                )

            if max_samples_per_split is not None:
                samples = samples[:max_samples_per_split]
            samples_by_dataset.setdefault(dataset, {})[split_suffix] = samples

    return samples_by_dataset, files_per_sample


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def transcode_clip(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    video_filter: str,
    preset: str,
    crf: int,
) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path.name}: {result.stderr.strip() or result.stdout}"
        )


def write_shard(
    *,
    zip_path: str,
    shard_path: str,
    samples: list[Sample],
    ffmpeg_bin: str,
    video_filter: str,
    preset: str,
    crf: int,
    write_json_metadata: bool,
) -> dict:
    shard_file = Path(shard_path)
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0

    with ZipFile(zip_path) as archive, tempfile.TemporaryDirectory(
        prefix="fsboard_wds_"
    ) as tmpdir, tarfile.open(shard_file, "w") as tar:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.mp4"
        output_path = tmpdir_path / "output.mp4"

        for sample in samples:
            input_path.write_bytes(archive.read(sample.zip_member))
            if output_path.exists():
                output_path.unlink()
            transcode_clip(
                ffmpeg_bin=ffmpeg_bin,
                input_path=input_path,
                output_path=output_path,
                video_filter=video_filter,
                preset=preset,
                crf=crf,
            )

            add_bytes_to_tar(tar, f"{sample.key}.mp4", output_path.read_bytes())
            add_bytes_to_tar(
                tar, f"{sample.key}.txt", (sample.transcript + "\n").encode("utf-8")
            )
            if write_json_metadata:
                metadata_bytes = json.dumps(sample.metadata, ensure_ascii=True).encode(
                    "utf-8"
                )
                add_bytes_to_tar(tar, f"{sample.key}.json", metadata_bytes)
            sample_count += 1

    return {
        "shard": str(shard_file),
        "samples": sample_count,
        "files": sample_count * (3 if write_json_metadata else 2),
    }


def main() -> None:
    args = parse_args()
    datasets_filter = set(args.datasets) if args.datasets else None

    samples_by_dataset, files_per_sample = build_samples(
        zip_path=args.zip_path,
        datasets_filter=datasets_filter,
        include_json_metadata=args.write_json_metadata,
        max_samples_per_split=args.max_samples_per_split,
    )

    samples_per_shard = max(1, args.max_files_per_shard // files_per_sample)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_jobs = []
    for dataset, splits in samples_by_dataset.items():
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split, samples in splits.items():
            split_dir = dataset_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for shard_index, shard_samples in enumerate(
                chunked(samples, samples_per_shard)
            ):
                shard_path = split_dir / f"{dataset}-{split}-{shard_index:06d}.tar"
                shard_jobs.append(
                    {
                        "zip_path": args.zip_path,
                        "shard_path": str(shard_path),
                        "samples": shard_samples,
                        "ffmpeg_bin": args.ffmpeg,
                        "video_filter": args.video_filter,
                        "preset": args.preset,
                        "crf": args.crf,
                        "write_json_metadata": args.write_json_metadata,
                    }
                )

    manifest = {
        "zip_path": args.zip_path,
        "workers": args.workers,
        "max_files_per_shard": args.max_files_per_shard,
        "files_per_sample": files_per_sample,
        "samples_per_shard": samples_per_shard,
        "datasets": {
            dataset: {
                split: {
                    "samples": len(samples),
                    "expected_shards": math.ceil(len(samples) / samples_per_shard)
                    if samples
                    else 0,
                }
                for split, samples in splits.items()
            }
            for dataset, splits in samples_by_dataset.items()
        },
        "shards": [],
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(write_shard, **job) for job in shard_jobs]
        for future in as_completed(futures):
            result = future.result()
            manifest["shards"].append(result)
            print(
                f"[done] {result['shard']} samples={result['samples']} files={result['files']}",
                flush=True,
            )

    manifest["shards"].sort(key=lambda item: item["shard"])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(manifest['shards'])} shard(s) to {output_dir} "
        f"with up to {samples_per_shard} samples per shard.",
        flush=True,
    )


if __name__ == "__main__":
    main()
