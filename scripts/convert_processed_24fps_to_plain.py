#!/usr/bin/env python3
"""
Convert the 24fps WebDataset collection under processed_24fps into the plain
format used by all_*_plain_v3.

Outputs:
- <output_root>/videos/...
- <output_root>/manifests/videos.csv
- <output_root>/manifests/paired_manifest.tsv
- <output_root>/manifests/videos.part-XXX.csv
- <output_root>/manifests/paired_manifest.part-XXX.tsv
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    split: str
    path: Path
    dataset_name: str
    language: str
    output_subdir: str | None = None
    recursive: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/mnt/data2/sign_language_24fps/processed_24fps"),
        help="Root directory containing the 24fps webdataset sources.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data2/sign_language_24fps/processed_24fps"),
        help="Root directory where all_{train,val,test}_plain_v3 will be created.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Worker processes per split.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rewriting existing extracted videos.",
    )
    parser.add_argument(
        "--limit-shards-per-source",
        type=int,
        default=None,
        help="Optional small trial mode: limit each source to this many shards.",
    )
    return parser.parse_args()


def build_source_specs(processed_root: Path) -> list[SourceSpec]:
    return [
        SourceSpec("train", processed_root / "YoutubeASL_processed", "youtubeasl", "asl"),
        SourceSpec("train", processed_root / "csl_news", "csl_wds", "csl"),
        SourceSpec("train", processed_root / "bobsl_24fps" / "train", "bobsl_wds", "bsl"),
        SourceSpec("val", processed_root / "bobsl_24fps" / "val", "bobsl_wds", "bsl"),
        SourceSpec("test", processed_root / "bobsl_24fps" / "test", "bobsl_wds", "bsl"),
        SourceSpec(
            "train",
            processed_root / "bobsl_24fps_manual" / "train",
            "bobsl_wds_manual",
            "bsl",
        ),
        SourceSpec(
            "val",
            processed_root / "bobsl_24fps_manual" / "val",
            "bobsl_wds_manual",
            "bsl",
        ),
        SourceSpec(
            "test",
            processed_root / "bobsl_24fps_manual" / "test",
            "bobsl_wds_manual",
            "bsl",
        ),
        SourceSpec("train", processed_root / "csl_daily_24fps" / "train.tar", "csl_daily", "csl"),
        SourceSpec("val", processed_root / "csl_daily_24fps" / "dev.tar", "csl_daily", "csl"),
        SourceSpec("test", processed_root / "csl_daily_24fps" / "test.tar", "csl_daily", "csl"),
        SourceSpec("train", processed_root / "how2sign_24fps" / "train", "how2sign", "asl"),
        SourceSpec("val", processed_root / "how2sign_24fps" / "val", "how2sign", "asl"),
        SourceSpec("test", processed_root / "how2sign_24fps" / "test", "how2sign", "asl"),
        SourceSpec("train", processed_root / "openasl_24fps" / "train", "openasl", "asl"),
        SourceSpec("val", processed_root / "openasl_24fps" / "val", "openasl", "asl"),
        SourceSpec("test", processed_root / "openasl_24fps" / "test", "openasl", "asl"),
        SourceSpec(
            "train",
            processed_root / "dailymoth-70h" / "train",
            "daily_moth",
            "asl",
            output_subdir="dailymoth_train",
        ),
        SourceSpec(
            "val",
            processed_root / "dailymoth-70h" / "val",
            "daily_moth",
            "asl",
            output_subdir="dailymoth_val",
        ),
        SourceSpec(
            "test",
            processed_root / "dailymoth-70h" / "test",
            "daily_moth",
            "asl",
            output_subdir="dailymotn_test",
        ),
    ]


def discover_shards(spec: SourceSpec, limit: int | None) -> list[dict[str, str]]:
    path = spec.path
    if path.is_file():
        shards = [path]
    elif spec.recursive:
        shards = sorted(p for p in path.rglob("*.tar") if p.is_file())
    else:
        shards = sorted(p for p in path.glob("*.tar") if p.is_file())

    if limit is not None:
        shards = shards[:limit]

    return [
        {
            "shard_path": str(shard),
            "dataset_name": spec.dataset_name,
            "language": spec.language,
            "output_subdir": spec.output_subdir or "",
        }
        for shard in shards
    ]


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


def extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path, skip_existing: bool) -> None:
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


def make_output_path(videos_root: Path, shard_path: Path, rel_path: Path, output_subdir: str) -> Path:
    if output_subdir:
        return videos_root / output_subdir / rel_path.name
    return videos_root / shard_path.stem / rel_path


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
        return f"{float(value):.6f}" if value else "nan"
    except Exception:
        return "nan"


def chunked(items: list[dict[str, str]], n_chunks: int) -> list[list[dict[str, str]]]:
    if n_chunks <= 1:
        return [items]
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def process_shards(
    worker_idx: int,
    shards: list[dict[str, str]],
    output_root: str,
    skip_existing: bool,
) -> dict[str, str | int]:
    output_root_path = Path(output_root)
    manifests_root = output_root_path / "manifests"
    videos_root = output_root_path / "videos"
    manifests_root.mkdir(parents=True, exist_ok=True)

    video_part_path = manifests_root / f"videos.part-{worker_idx:03d}.csv"
    paired_part_path = manifests_root / f"paired_manifest.part-{worker_idx:03d}.tsv"

    extracted_videos = 0

    with open(video_part_path, "w", newline="") as video_csv, open(
        paired_part_path, "w", newline=""
    ) as paired_csv:
        video_writer = csv.writer(video_csv, delimiter=" ")
        paired_writer = csv.writer(paired_csv, delimiter="\t")

        for local_idx, shard_info in enumerate(shards, start=1):
            shard_path = Path(shard_info["shard_path"])
            dataset_name = str(shard_info["dataset_name"])
            language = str(shard_info["language"])
            output_subdir = str(shard_info["output_subdir"])

            print(
                f"[worker {worker_idx:03d}] [{local_idx}/{len(shards)}] {shard_path}",
                flush=True,
            )

            with tarfile.open(shard_path, "r") as tar:
                grouped = group_members(tar.getmembers())
                for pair in grouped.values():
                    mp4_member = pair.get(".mp4")
                    txt_member = pair.get(".txt")
                    if mp4_member is None:
                        continue

                    rel_path = safe_relative_member_path(mp4_member.name)
                    video_dst = make_output_path(videos_root, shard_path, rel_path, output_subdir)
                    extract_member(tar, mp4_member, video_dst, skip_existing)
                    duration_seconds = probe_duration_seconds(video_dst)
                    transcript_text = read_text_member(tar, txt_member) if txt_member else ""

                    video_writer.writerow(
                        [str(video_dst.resolve()), 0, duration_seconds, dataset_name, language]
                    )
                    paired_writer.writerow(
                        [
                            str(video_dst.resolve()),
                            duration_seconds,
                            dataset_name,
                            language,
                            transcript_text,
                        ]
                    )
                    extracted_videos += 1

    return {
        "video_part_path": str(video_part_path),
        "paired_part_path": str(paired_part_path),
        "extracted_videos": extracted_videos,
    }


def merge_parts(part_paths: list[Path], merged_path: Path) -> None:
    with open(merged_path, "w", newline="") as out:
        for part_path in sorted(part_paths):
            if not part_path.exists():
                continue
            with open(part_path, "r", newline="") as part:
                for line in part:
                    out.write(line)


def process_split(
    split: str,
    shard_infos: list[dict[str, str]],
    output_root: Path,
    num_workers: int,
    skip_existing: bool,
) -> None:
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    if not shard_infos:
        print(f"[{split}] no shards found, skipping")
        return

    shard_groups = chunked(shard_infos, min(max(1, num_workers), len(shard_infos)))
    results = []

    if len(shard_groups) == 1:
        results.append(
            process_shards(
                worker_idx=0,
                shards=shard_groups[0],
                output_root=str(output_root),
                skip_existing=skip_existing,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=len(shard_groups)) as executor:
            futures = [
                executor.submit(
                    process_shards,
                    worker_idx=i,
                    shards=group,
                    output_root=str(output_root),
                    skip_existing=skip_existing,
                )
                for i, group in enumerate(shard_groups)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    merge_parts([Path(r["video_part_path"]) for r in results], manifests_root / "videos.csv")
    merge_parts(
        [Path(r["paired_part_path"]) for r in results],
        manifests_root / "paired_manifest.tsv",
    )

    print(
        f"[{split}] finished"
        f" shards={len(shard_infos)}"
        f" videos={sum(int(r['extracted_videos']) for r in results)}"
        f" output={output_root}"
    )

def main() -> int:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()

    if not processed_root.is_dir():
        print(f"Missing processed root: {processed_root}", file=sys.stderr)
        return 1

    specs = [spec for spec in build_source_specs(processed_root) if spec.path.exists()]
    split_to_shards: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for spec in specs:
        split_to_shards[spec.split].extend(discover_shards(spec, args.limit_shards_per_source))

    for split, shard_infos in split_to_shards.items():
        split_output_root = output_root / f"all_{split}_plain_v3"
        process_split(
            split=split,
            shard_infos=sorted(shard_infos, key=lambda item: item["shard_path"]),
            output_root=split_output_root,
            num_workers=args.num_workers,
            skip_existing=args.skip_existing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
