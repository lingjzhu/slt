#!/usr/bin/env python3
"""
Extract WebDataset-style tar shards containing paired .mp4/.txt samples into
plain files and build manifests for both V-JEPA pretraining and later SLT.

Generated outputs:
- manifests/videos.csv
    /abs/path/to/video.mp4 0 duration_seconds dataset_name language
- manifests/paired_manifest.csv
    /abs/path/to/video.mp4 duration_seconds dataset_name language transcript_text

The first two columns of videos.csv remain compatible with the current
V-JEPA VideoDataset loader. The paired manifest keeps the actual transcript
text inline for downstream sign language translation training.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="Input directory containing tar shards. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where extracted files and manifests will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search each input directory for tar files.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Optional limit for a smaller trial run after shard discovery.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite files that already exist.",
    )
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Extract only .mp4 files. paired_manifest.csv will still be written when transcript text exists in tar.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes. Use >1 for parallel shard extraction.",
    )
    return parser.parse_args()


def infer_dataset_name(input_dir: Path) -> str:
    path_str = str(input_dir).lower()
    if "bobsl/wds_manual" in path_str:
        return "bobsl_wds_manual"
    if "bobsl/wds" in path_str:
        return "bobsl_wds"
    if "how2sign" in path_str:
        return "how2sign"
    if "openasl" in path_str:
        return "openasl"
    if "youtubeasl" in path_str:
        return "youtubeasl"
    if "csl_daily" in path_str:
        return "csl_daily"
    if "csl_wds" in path_str:
        return "csl_wds"
    if "daily_moth" in path_str:
        return "daily_moth"
    if "bobsl" in path_str:
        return "bobsl"
    return input_dir.name


def infer_language(dataset_name: str) -> str:
    if dataset_name in {"how2sign", "openasl", "youtubeasl", "daily_moth"}:
        return "asl"
    if dataset_name in {"csl_daily", "csl_wds"}:
        return "csl"
    if dataset_name in {"bobsl", "bobsl_wds", "bobsl_wds_manual"}:
        return "bsl"
    return "unknown"


def discover_shards(
    input_dirs: list[Path], recursive: bool, limit: int | None
) -> list[dict[str, str]]:
    shards: list[dict[str, str]] = []
    for input_dir in input_dirs:
        dataset_name = infer_dataset_name(input_dir)
        language = infer_language(dataset_name)
        if recursive:
            found = sorted(p for p in input_dir.rglob("*.tar") if p.is_file())
        else:
            found = sorted(
                p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".tar"
            )
        shards.extend(
            {
                "shard_path": str(path),
                "dataset_name": dataset_name,
                "language": language,
            }
            for path in found
        )

    shards = sorted(shards, key=lambda item: item["shard_path"])
    if limit is not None:
        shards = shards[:limit]
    return shards


def safe_relative_member_path(member_name: str) -> Path:
    path = Path(member_name)
    if path.is_absolute():
        raise ValueError(f"Refusing absolute member path: {member_name}")

    clean_parts = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"Refusing unsafe member path: {member_name}")
        clean_parts.append(part)

    if not clean_parts:
        raise ValueError(f"Refusing empty member path: {member_name}")
    return Path(*clean_parts)


def extract_member(
    tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path, skip_existing: bool
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and destination.exists():
        return

    fileobj = tar.extractfile(member)
    if fileobj is None:
        return

    with fileobj, open(destination, "wb") as out:
        while True:
            chunk = fileobj.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def read_text_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    fileobj = tar.extractfile(member)
    if fileobj is None:
        return ""
    with fileobj:
        return fileobj.read().decode("utf-8", errors="replace").strip().replace("\n", " ")


def make_output_path(
    videos_root: Path, texts_root: Path, shard_path: Path, rel_path: Path, suffix: str
) -> Path:
    stem_prefix = shard_path.stem
    if suffix == ".mp4":
        return videos_root / stem_prefix / rel_path
    return texts_root / stem_prefix / rel_path


def probe_duration_seconds(path: Path) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        value = result.stdout.strip()
        if not value:
            return "nan"
        return f"{float(value):.6f}"
    except Exception:
        return "nan"


def chunked(items: list[dict[str, str]], n_chunks: int) -> list[list[dict[str, str]]]:
    if n_chunks <= 1:
        return [items]
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def group_members(members: list[tarfile.TarInfo]) -> dict[str, dict[str, tarfile.TarInfo]]:
    grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in members:
        if not member.isfile():
            continue
        rel_path = safe_relative_member_path(member.name)
        suffix = rel_path.suffix.lower()
        if suffix not in (".mp4", ".txt"):
            continue
        stem = str(rel_path.with_suffix(""))
        grouped.setdefault(stem, {})[suffix] = member
    return grouped


def process_shards(
    worker_idx: int,
    shards: list[dict[str, str]],
    output_root: str,
    skip_existing: bool,
    videos_only: bool,
) -> dict:
    output_root_path = Path(output_root)
    videos_root = output_root_path / "videos"
    texts_root = output_root_path / "texts"
    manifests_root = output_root_path / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    video_part_path = manifests_root / f"videos.part-{worker_idx:03d}.csv"
    paired_part_path = manifests_root / f"paired_manifest.part-{worker_idx:03d}.tsv"

    extracted_videos = 0
    extracted_texts = 0

    with open(video_part_path, "w", newline="") as video_csv, open(
        paired_part_path, "w", newline=""
    ) as paired_csv:
        video_writer = csv.writer(video_csv, delimiter=" ")
        paired_writer = csv.writer(paired_csv, delimiter="\t")

        for local_idx, shard_info in enumerate(shards, start=1):
            shard_path = Path(shard_info["shard_path"])
            dataset_name = shard_info["dataset_name"]
            language = shard_info["language"]
            print(
                f"[worker {worker_idx:03d}] [{local_idx}/{len(shards)}] {shard_path}",
                flush=True,
            )
            with tarfile.open(shard_path, "r") as tar:
                grouped = group_members(tar.getmembers())
                for _, pair in grouped.items():
                    mp4_member = pair.get(".mp4")
                    txt_member = pair.get(".txt")
                    if mp4_member is None:
                        continue

                    rel_path = safe_relative_member_path(mp4_member.name)
                    video_dst = make_output_path(
                        videos_root, texts_root, shard_path, rel_path, ".mp4"
                    )
                    extract_member(tar, mp4_member, video_dst, skip_existing)
                    duration_seconds = probe_duration_seconds(video_dst)
                    video_writer.writerow(
                        [
                            str(video_dst.resolve()),
                            0,
                            duration_seconds,
                            dataset_name,
                            language,
                        ]
                    )
                    extracted_videos += 1

                    transcript_text = ""
                    if txt_member is not None:
                        transcript_text = read_text_member(tar, txt_member)
                        extracted_texts += 1
                        if not videos_only:
                            txt_rel_path = safe_relative_member_path(txt_member.name)
                            text_dst = make_output_path(
                                videos_root, texts_root, shard_path, txt_rel_path, ".txt"
                            )
                            extract_member(tar, txt_member, text_dst, skip_existing)

                    paired_writer.writerow(
                        [
                            str(video_dst.resolve()),
                            duration_seconds,
                            dataset_name,
                            language,
                            transcript_text,
                        ]
                    )

    return {
        "worker_idx": worker_idx,
        "video_part_path": str(video_part_path),
        "paired_part_path": str(paired_part_path),
        "extracted_videos": extracted_videos,
        "extracted_texts": extracted_texts,
    }


def merge_parts(part_paths: list[Path], merged_path: Path) -> None:
    with open(merged_path, "w", newline="") as out:
        for part_path in sorted(part_paths):
            if not part_path.exists():
                continue
            with open(part_path, "r", newline="") as part:
                for line in part:
                    out.write(line)


def main() -> int:
    args = parse_args()
    input_dirs = [p.resolve() for p in args.input_dir]
    output_root = args.output_root.resolve()

    for input_dir in input_dirs:
        if not input_dir.is_dir():
            print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
            return 1

    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(input_dirs, args.recursive, args.limit_shards)
    if not shards:
        print("No .tar shards found in the provided input directories.", file=sys.stderr)
        return 1

    num_workers = max(1, args.num_workers)
    shard_groups = chunked(shards, min(num_workers, len(shards)))

    worker_results = []
    if len(shard_groups) == 1:
        worker_results.append(
            process_shards(
                worker_idx=0,
                shards=shard_groups[0],
                output_root=str(output_root),
                skip_existing=args.skip_existing,
                videos_only=args.videos_only,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=len(shard_groups)) as executor:
            futures = [
                executor.submit(
                    process_shards,
                    worker_idx=i,
                    shards=shard_group,
                    output_root=str(output_root),
                    skip_existing=args.skip_existing,
                    videos_only=args.videos_only,
                )
                for i, shard_group in enumerate(shard_groups)
            ]
            for fut in as_completed(futures):
                worker_results.append(fut.result())

    merge_parts(
        [Path(r["video_part_path"]) for r in worker_results],
        manifests_root / "videos.csv",
    )
    merge_parts(
        [Path(r["paired_part_path"]) for r in worker_results],
        manifests_root / "paired_manifest.tsv",
    )

    extracted_videos = sum(r["extracted_videos"] for r in worker_results)
    extracted_texts = sum(r["extracted_texts"] for r in worker_results)

    print(
        "Finished:"
        f" shards={len(shards)}"
        f" workers={len(shard_groups)}"
        f" videos={extracted_videos}"
        f" texts={extracted_texts}"
        f" video_manifest={manifests_root / 'videos.csv'}"
        f" paired_manifest={manifests_root / 'paired_manifest.tsv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
