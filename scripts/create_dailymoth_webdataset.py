#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WebDataset tar shards for Daily Moth from extracted videos and manifests."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the extracted dailymoth-70h-24fps directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to the output dailymoth-70h WebDataset directory.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per tar shard.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def extract_manifests(manifest_tar: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="dailymoth_manifests_"))
    with tarfile.open(manifest_tar, "r:gz") as tar:
        tar.extractall(tmpdir)
    return tmpdir / "dailymoth-70h" / "manifests"


def build_video_index(source_root: Path) -> dict[str, Path]:
    return {path.name: path for path in source_root.rglob("*.mp4")}


def add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def iter_split_rows(path: Path) -> list[tuple[str, str, str]]:
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [tuple(row) for row in reader if row]
    return rows


def write_split(
    split_name: str,
    rows: list[tuple[str, str, str]],
    video_index: dict[str, Path],
    output_root: Path,
    shard_size: int,
) -> None:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = -1
    tar: tarfile.TarFile | None = None

    try:
        for sample_idx, (video_name, _duration, caption) in enumerate(rows):
            if sample_idx % shard_size == 0:
                if tar is not None:
                    tar.close()
                shard_idx += 1
                shard_path = split_dir / f"{shard_idx:05d}.tar"
                tar = tarfile.open(shard_path, "w")

            assert tar is not None
            video_path = video_index.get(video_name)
            if video_path is None:
                raise FileNotFoundError(f"Missing video for manifest row: {video_name}")

            base = Path(video_name).stem
            tar.add(video_path, arcname=f"./{base}.mp4", recursive=False)
            add_bytes(tar, f"./{base}.txt", caption.encode("utf-8"))
    finally:
        if tar is not None:
            tar.close()


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_tar = source_root / "manifests.tar.gz"

    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not manifest_tar.is_file():
        raise FileNotFoundError(manifest_tar)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Pass --overwrite to reuse it."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    manifests_dir = extract_manifests(manifest_tar)
    video_index = build_video_index(source_root)

    for split_name in ["train", "val", "test"]:
        rows = iter_split_rows(manifests_dir / f"{split_name}.tsv")
        print(f"{split_name}: {len(rows)} rows")
        write_split(split_name, rows, video_index, output_root, args.shard_size)

    print(f"Created WebDataset shards under {output_root}")


if __name__ == "__main__":
    main()
