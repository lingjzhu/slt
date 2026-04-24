#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("/home/slime-base/projects/jian/islr/webdataset")
PROJECT_ROOT = Path("/home/slime-base/projects/jian/islr")
VIDEO_ROOT = PROJECT_ROOT / "data" / "preprocessed_videos_full"


def sanitize_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


@dataclass
class Sample:
    key: str
    video_bytes: bytes
    metadata: dict


class TarShardWriter:
    def __init__(self, output_dir: Path, split: str, max_samples_per_shard: int = 1000):
        self.output_dir = output_dir
        self.split = split
        self.max_samples_per_shard = max_samples_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tar: tarfile.TarFile | None = None
        self._shard_index = -1
        self._samples_in_shard = 0
        self.shards: list[str] = []
        self.count = 0

    def _open_next_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
        self._shard_index += 1
        shard_name = f"{self.split}-{self._shard_index:06d}.tar"
        self.shards.append(shard_name)
        self._tar = tarfile.open(self.output_dir / shard_name, mode="w")
        self._samples_in_shard = 0

    def write(self, sample: Sample) -> None:
        if self._tar is None or self._samples_in_shard >= self.max_samples_per_shard:
            self._open_next_shard()

        assert self._tar is not None
        base_name = sample.key

        mp4_info = tarfile.TarInfo(name=f"{base_name}.mp4")
        mp4_info.size = len(sample.video_bytes)
        self._tar.addfile(mp4_info, io.BytesIO(sample.video_bytes))

        metadata_bytes = json.dumps(sample.metadata, ensure_ascii=True, sort_keys=True).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{base_name}.json")
        json_info.size = len(metadata_bytes)
        self._tar.addfile(json_info, io.BytesIO(metadata_bytes))

        self._samples_in_shard += 1
        self.count += 1

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


def read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def write_json(path: Path, payload: dict, *, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def process_asl_citizen(zip_path: Path, output_root: Path, max_samples_per_shard: int) -> dict:
    dataset_name = "asl_citizen"
    dataset_root = output_root / dataset_name
    video_prefix = "ASL_Citizen/videos/"
    split_members = {
        "train": "ASL_Citizen/splits/train.csv",
        "val": "ASL_Citizen/splits/val.csv",
        "test": "ASL_Citizen/splits/test.csv",
    }

    summary: dict[str, dict] = {"dataset": dataset_name, "source_zip": str(zip_path), "splits": {}}

    with zipfile.ZipFile(zip_path) as zf:
        rows_by_split = {split: read_csv_from_zip(zf, member) for split, member in split_members.items()}

        all_glosses = sorted({row["Gloss"] for rows in rows_by_split.values() for row in rows})
        gloss_to_label = {gloss: idx for idx, gloss in enumerate(all_glosses)}
        write_json(dataset_root / "label_map.json", {"label_field": "gloss", "labels": gloss_to_label})

        for split, rows in rows_by_split.items():
            writer = TarShardWriter(dataset_root / split, split=split, max_samples_per_shard=max_samples_per_shard)
            participant_counts = Counter()

            for row in rows:
                video_name = row["Video file"]
                video_bytes = zf.read(video_prefix + video_name)
                gloss = row["Gloss"]
                participant_id = row["Participant ID"]
                sample_key = sanitize_key(Path(video_name).stem)

                metadata = {
                    "dataset": dataset_name,
                    "split": split,
                    "key": sample_key,
                    "video_file": video_name,
                    "participant_id": participant_id,
                    "gloss": gloss,
                    "label": gloss_to_label[gloss],
                    "asl_lex_code": row["ASL-LEX Code"],
                    "source_archive": zip_path.name,
                }
                writer.write(Sample(key=sample_key, video_bytes=video_bytes, metadata=metadata))
                participant_counts[participant_id] += 1

            writer.close()
            split_summary = {
                "num_samples": writer.count,
                "num_shards": len(writer.shards),
                "shards": writer.shards,
                "participants": dict(sorted(participant_counts.items())),
            }
            summary["splits"][split] = split_summary
            write_json(dataset_root / f"{split}_manifest.json", split_summary)

    write_json(dataset_root / "summary.json", summary)
    return summary


def build_wlasl_metadata_lookup(zf: zipfile.ZipFile) -> dict[str, dict]:
    entries = json.loads(zf.read("WLASL_v0.3.json"))
    lookup: dict[str, dict] = {}
    for class_index, entry in enumerate(entries):
        gloss = entry["gloss"]
        for instance in entry["instances"]:
            video_id = instance["video_id"]
            lookup[video_id] = {
                "gloss": gloss,
                "full_class_index": class_index,
                "instance": instance,
            }
    return lookup


def process_wlasl(zip_path: Path, output_root: Path, max_samples_per_shard: int) -> dict:
    dataset_name = "wlasl_processed"
    dataset_root = output_root / dataset_name
    benchmark_members = ["nslt_100.json", "nslt_300.json", "nslt_1000.json", "nslt_2000.json"]

    summary: dict[str, dict] = {"dataset": dataset_name, "source_zip": str(zip_path), "benchmarks": {}}

    with zipfile.ZipFile(zip_path) as zf:
        metadata_lookup = build_wlasl_metadata_lookup(zf)
        zip_members = set(zf.namelist())

        missing_list = sorted(
            line.strip()
            for line in zf.read("missing.txt").decode("utf-8").splitlines()
            if line.strip()
        )
        write_json(dataset_root / "missing_videos.json", {"missing_videos": missing_list})

        for member in benchmark_members:
            benchmark_name = Path(member).stem
            annotations: dict[str, dict] = json.loads(zf.read(member))
            benchmark_root = dataset_root / benchmark_name
            split_writers = {
                split: TarShardWriter(benchmark_root / split, split=split, max_samples_per_shard=max_samples_per_shard)
                for split in ("train", "val", "test")
            }
            split_counts = Counter()
            missing_from_zip: list[str] = []

            for video_id, benchmark_meta in annotations.items():
                split = benchmark_meta["subset"]
                if split not in split_writers:
                    continue
                video_member = f"videos/{video_id}.mp4"
                if video_member not in zip_members:
                    missing_from_zip.append(video_id)
                    continue

                video_bytes = zf.read(video_member)
                extra = metadata_lookup.get(video_id, {})
                instance = extra.get("instance", {})
                class_id, _, class_count = benchmark_meta["action"]
                sample_key = sanitize_key(f"{benchmark_name}_{video_id}")
                metadata = {
                    "dataset": dataset_name,
                    "benchmark": benchmark_name,
                    "split": split,
                    "key": sample_key,
                    "video_id": video_id,
                    "video_file": f"{video_id}.mp4",
                    "label": class_id,
                    "num_classes": class_count,
                    "action": benchmark_meta["action"],
                    "gloss": extra.get("gloss"),
                    "full_class_index": extra.get("full_class_index"),
                    "fps": instance.get("fps"),
                    "frame_start": instance.get("frame_start"),
                    "frame_end": instance.get("frame_end"),
                    "bbox": instance.get("bbox"),
                    "signer_id": instance.get("signer_id"),
                    "source": instance.get("source"),
                    "variation_id": instance.get("variation_id"),
                    "instance_id": instance.get("instance_id"),
                    "original_split": instance.get("split"),
                    "url": instance.get("url"),
                    "source_archive": zip_path.name,
                }
                split_writers[split].write(
                    Sample(key=sample_key, video_bytes=video_bytes, metadata=metadata)
                )
                split_counts[split] += 1

            for writer in split_writers.values():
                writer.close()

            benchmark_summary = {
                "num_samples": int(sum(split_counts.values())),
                "splits": {
                    split: {
                        "num_samples": split_writers[split].count,
                        "num_shards": len(split_writers[split].shards),
                        "shards": split_writers[split].shards,
                    }
                    for split in ("train", "val", "test")
                },
                "missing_from_zip": sorted(missing_from_zip),
            }
            summary["benchmarks"][benchmark_name] = benchmark_summary
            write_json(benchmark_root / "summary.json", benchmark_summary)

    write_json(dataset_root / "summary.json", summary)
    return summary


def process_wlasl_hub(output_root: Path, max_samples_per_shard: int) -> dict:
    dataset_name = "wlasl"
    video_glob = "start_kit/raw_videos_mp4/*.zip"
    dataset_root = output_root / dataset_name
    hub_snapshot = Path(
        snapshot_download(
            repo_id="aipieces/WLASL",
            repo_type="dataset",
            allow_patterns=["WLASL_v0.3.json", video_glob],
        )
    )

    metadata_path = hub_snapshot / "WLASL_v0.3.json"
    shard_paths = sorted((hub_snapshot / "start_kit" / "raw_videos_mp4").glob("*.zip"))
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file in snapshot: {metadata_path}")
    if not shard_paths:
        raise FileNotFoundError(f"No video shards found in snapshot: {hub_snapshot / 'videos'}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    gloss_to_label = {entry["gloss"]: idx for idx, entry in enumerate(entries)}
    write_json(dataset_root / "label_map.json", {"label_field": "gloss", "labels": gloss_to_label})

    zip_handles = [zipfile.ZipFile(path) for path in shard_paths]
    video_index: dict[str, tuple[zipfile.ZipFile, str]] = {}
    for zf in zip_handles:
        for member in zf.namelist():
            if member.endswith(".mp4"):
                video_index[Path(member).stem] = (zf, member)

    split_writers = {
        split: TarShardWriter(dataset_root / split, split=split, max_samples_per_shard=max_samples_per_shard)
        for split in ("train", "val", "test")
    }
    split_counts = Counter()
    missing_from_shards: list[str] = []

    try:
        for class_index, entry in enumerate(entries):
            gloss = entry["gloss"]
            for instance in entry["instances"]:
                split = instance["split"]
                if split not in split_writers:
                    continue
                video_id = instance["video_id"]
                zipped = video_index.get(video_id)
                if zipped is None:
                    missing_from_shards.append(video_id)
                    continue
                zf, member = zipped
                video_bytes = zf.read(member)
                sample_key = sanitize_key(video_id)
                metadata = {
                    "dataset": dataset_name,
                    "split": split,
                    "key": sample_key,
                    "video_id": video_id,
                    "video_file": f"{video_id}.mp4",
                    "gloss": gloss,
                    "label": class_index,
                    "bbox": instance.get("bbox"),
                    "fps": instance.get("fps"),
                    "frame_start": instance.get("frame_start"),
                    "frame_end": instance.get("frame_end"),
                    "instance_id": instance.get("instance_id"),
                    "signer_id": instance.get("signer_id"),
                    "source": instance.get("source"),
                    "url": instance.get("url"),
                    "variation_id": instance.get("variation_id"),
                    "source_repository": "aipieces/WLASL",
                    "source_snapshot": hub_snapshot.name,
                }
                split_writers[split].write(
                    Sample(key=sample_key, video_bytes=video_bytes, metadata=metadata)
                )
                split_counts[split] += 1
    finally:
        for writer in split_writers.values():
            writer.close()
        for zf in zip_handles:
            zf.close()

    summary = {
        "dataset": dataset_name,
        "source_repository": "aipieces/WLASL",
        "source_snapshot": hub_snapshot.name,
        "num_classes": len(entries),
        "num_video_shards": len(shard_paths),
        "source_video_glob": video_glob,
        "splits": {
            split: {
                "num_samples": split_writers[split].count,
                "num_shards": len(split_writers[split].shards),
                "shards": split_writers[split].shards,
            }
            for split in ("train", "val", "test")
        },
        "missing_from_shards": sorted(set(missing_from_shards)),
    }
    write_json(dataset_root / "summary.json", summary)
    return summary


def load_wlasl_official_metadata() -> list[dict]:
    metadata_path = Path(
        hf_hub_download(
            repo_id="iMaryam2/WLASL_Updated",
            repo_type="dataset",
            filename="WLASL_v0.3.json",
        )
    )
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_wlasl2000_local(zip_path: Path, output_root: Path, max_samples_per_shard: int) -> dict:
    dataset_name = "wlasl"
    dataset_root = output_root / dataset_name
    entries = load_wlasl_official_metadata()

    gloss_to_label = {entry["gloss"]: idx for idx, entry in enumerate(entries)}
    write_json(dataset_root / "label_map.json", {"label_field": "gloss", "labels": gloss_to_label})

    split_writers = {
        split: TarShardWriter(dataset_root / split, split=split, max_samples_per_shard=max_samples_per_shard)
        for split in ("train", "val", "test")
    }
    split_counts = Counter()
    extra_files: list[str] = []
    missing_from_zip: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        zip_members = {Path(name).stem: name for name in zf.namelist() if name.endswith(".mp4")}
        official_ids = {
            instance["video_id"]
            for entry in entries
            for instance in entry["instances"]
        }
        extra_files = sorted(video_id for video_id in zip_members if video_id not in official_ids)

        try:
            for class_index, entry in enumerate(entries):
                gloss = entry["gloss"]
                for instance in entry["instances"]:
                    split = instance["split"]
                    if split not in split_writers:
                        continue
                    video_id = instance["video_id"]
                    member = zip_members.get(video_id)
                    if member is None:
                        missing_from_zip.append(video_id)
                        continue

                    video_bytes = zf.read(member)
                    sample_key = sanitize_key(video_id)
                    metadata = {
                        "dataset": dataset_name,
                        "split": split,
                        "key": sample_key,
                        "video_id": video_id,
                        "video_file": f"{video_id}.mp4",
                        "gloss": gloss,
                        "label": class_index,
                        "bbox": instance.get("bbox"),
                        "fps": instance.get("fps"),
                        "frame_start": instance.get("frame_start"),
                        "frame_end": instance.get("frame_end"),
                        "instance_id": instance.get("instance_id"),
                        "signer_id": instance.get("signer_id"),
                        "source": instance.get("source"),
                        "url": instance.get("url"),
                        "variation_id": instance.get("variation_id"),
                        "source_archive": zip_path.name,
                        "metadata_source": "iMaryam2/WLASL_Updated:WLASL_v0.3.json",
                    }
                    split_writers[split].write(
                        Sample(key=sample_key, video_bytes=video_bytes, metadata=metadata)
                    )
                    split_counts[split] += 1
        finally:
            for writer in split_writers.values():
                writer.close()

    summary = {
        "dataset": dataset_name,
        "source_archive": str(zip_path),
        "metadata_source": "iMaryam2/WLASL_Updated:WLASL_v0.3.json",
        "num_classes": len(entries),
        "splits": {
            split: {
                "num_samples": split_writers[split].count,
                "num_shards": len(split_writers[split].shards),
                "shards": split_writers[split].shards,
            }
            for split in ("train", "val", "test")
        },
        "missing_from_zip": sorted(set(missing_from_zip)),
        "extra_files_ignored": extra_files,
    }
    write_json(dataset_root / "summary.json", summary)
    return summary


def resolve_preprocessed_video_path(raw_path: object, view_name: str) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None

    candidate = Path(raw_path)
    fallback_candidates = [
        candidate,
        PROJECT_ROOT / candidate.name,
        VIDEO_ROOT / view_name / candidate.name,
        Path(str(candidate).replace("/preprocessed_videos_full/", "/data/preprocessed_videos_full/")),
        Path(str(candidate).replace("/preprocessed_videos/", "/data/preprocessed_videos_full/")),
    ]
    for fallback in fallback_candidates:
        if fallback.exists():
            return fallback
    return None


def _write_preprocessed_split(
    dataframe: pd.DataFrame,
    split: str,
    writer: TarShardWriter,
    label_map: dict[int, int],
    source_csv: Path,
) -> tuple[Counter, list[dict[str, str]]]:
    view_counts = Counter()
    missing_paths: list[dict[str, str]] = []

    for _, row in dataframe.iterrows():
        raw_label = int(row["id"])
        for view_name, column_name in (("front", "filepath_front"), ("left", "filepath_left")):
            resolved_path = resolve_preprocessed_video_path(row.get(column_name), view_name=view_name)
            if resolved_path is None:
                raw_value = row.get(column_name)
                if isinstance(raw_value, str) and raw_value:
                    missing_paths.append({"view": view_name, "path": raw_value, "id": str(raw_label)})
                continue

            with resolved_path.open("rb") as handle:
                video_bytes = handle.read()

            sample_key = sanitize_key(f"{row['id']}_{view_name}_{resolved_path.stem}")
            metadata = {
                "dataset": "preprocessed_full",
                "split": split,
                "key": sample_key,
                "label": label_map[raw_label],
                "raw_label": raw_label,
                "transcription": str(row.get("transcription", "")),
                "id": str(row["id"]).zfill(4),
                "view": view_name,
                "video_file": resolved_path.name,
                "resolved_path": str(resolved_path),
                "filepath_front": None if pd.isna(row.get("filepath_front")) else str(row.get("filepath_front")),
                "filepath_left": None if pd.isna(row.get("filepath_left")) else str(row.get("filepath_left")),
                "duration": None if pd.isna(row.get("duration")) else float(row.get("duration")),
                "total_frames": None if pd.isna(row.get("total_frames")) else int(row.get("total_frames")),
                "source_csv": str(source_csv),
            }
            writer.write(Sample(key=sample_key, video_bytes=video_bytes, metadata=metadata))
            view_counts[view_name] += 1

    return view_counts, missing_paths


def process_preprocessed_csv(csv_path: Path, output_root: Path, max_samples_per_shard: int) -> dict:
    dataset_name = "preprocessed_full"
    dataset_root = output_root / dataset_name
    dataframe = pd.read_csv(csv_path)

    raw_ids = sorted(int(label_id) for label_id in dataframe["id"].unique())
    label_map = {label_id: index for index, label_id in enumerate(raw_ids)}
    transcription_map = {
        int(label_id): str(group["transcription"].iloc[0])
        for label_id, group in dataframe.groupby("id", sort=True)
    }
    write_json(
        dataset_root / "label_map.json",
        {
            "label_field": "id",
            "labels": {
                str(label_id): {
                    "index": label_map[label_id],
                    "transcription": transcription_map[label_id],
                }
                for label_id in raw_ids
            },
        },
        ensure_ascii=False,
    )

    writer = TarShardWriter(dataset_root / "all", split="all", max_samples_per_shard=max_samples_per_shard)
    view_counts = Counter()
    missing_paths: list[dict[str, str]] = []

    try:
        for _, row in dataframe.iterrows():
            raw_label = int(row["id"])
            for view_name, column_name in (("front", "filepath_front"), ("left", "filepath_left")):
                resolved_path = resolve_preprocessed_video_path(row.get(column_name), view_name=view_name)
                if resolved_path is None:
                    raw_value = row.get(column_name)
                    if isinstance(raw_value, str) and raw_value:
                        missing_paths.append({"view": view_name, "path": raw_value, "id": str(raw_label)})
                    continue

                with resolved_path.open("rb") as handle:
                    video_bytes = handle.read()

                sample_key = sanitize_key(f"{row['id']}_{view_name}_{resolved_path.stem}")
                metadata = {
                    "dataset": dataset_name,
                    "split": "all",
                    "key": sample_key,
                    "label": label_map[raw_label],
                    "raw_label": raw_label,
                    "transcription": str(row.get("transcription", "")),
                    "id": str(row["id"]).zfill(4),
                    "view": view_name,
                    "video_file": resolved_path.name,
                    "resolved_path": str(resolved_path),
                    "filepath_front": None if pd.isna(row.get("filepath_front")) else str(row.get("filepath_front")),
                    "filepath_left": None if pd.isna(row.get("filepath_left")) else str(row.get("filepath_left")),
                    "duration": None if pd.isna(row.get("duration")) else float(row.get("duration")),
                    "total_frames": None if pd.isna(row.get("total_frames")) else int(row.get("total_frames")),
                    "source_csv": str(csv_path),
                }
                writer.write(Sample(key=sample_key, video_bytes=video_bytes, metadata=metadata))
                view_counts[view_name] += 1
    finally:
        writer.close()

    summary = {
        "dataset": dataset_name,
        "source_csv": str(csv_path),
        "num_classes": len(label_map),
        "splits": {
            "all": {
                "num_samples": writer.count,
                "num_shards": len(writer.shards),
                "shards": writer.shards,
            }
        },
        "view_counts": dict(sorted(view_counts.items())),
        "missing_paths": missing_paths,
    }
    write_json(dataset_root / "summary.json", summary)
    return summary


def process_preprocessed_split_csvs(
    train_csv_path: Path,
    test_csv_path: Path,
    output_root: Path,
    max_samples_per_shard: int,
) -> dict:
    dataset_name = "preprocessed_full"
    dataset_root = output_root / dataset_name
    train_dataframe = pd.read_csv(train_csv_path)
    test_dataframe = pd.read_csv(test_csv_path)

    raw_ids = sorted({int(label_id) for label_id in train_dataframe["id"].unique()} | {int(label_id) for label_id in test_dataframe["id"].unique()})
    label_map = {label_id: index for index, label_id in enumerate(raw_ids)}
    combined_dataframe = pd.concat([train_dataframe, test_dataframe], ignore_index=True)
    transcription_map = {
        int(label_id): str(group["transcription"].iloc[0])
        for label_id, group in combined_dataframe.groupby("id", sort=True)
    }
    write_json(
        dataset_root / "label_map.json",
        {
            "label_field": "id",
            "labels": {
                str(label_id): {
                    "index": label_map[label_id],
                    "transcription": transcription_map[label_id],
                }
                for label_id in raw_ids
            },
        },
        ensure_ascii=False,
    )

    summaries: dict[str, dict] = {}
    total_missing_paths: list[dict[str, str]] = []
    combined_view_counts = Counter()

    for split, dataframe, source_csv in (
        ("train", train_dataframe, train_csv_path),
        ("test", test_dataframe, test_csv_path),
    ):
        writer = TarShardWriter(dataset_root / split, split=split, max_samples_per_shard=max_samples_per_shard)
        try:
            view_counts, missing_paths = _write_preprocessed_split(
                dataframe=dataframe,
                split=split,
                writer=writer,
                label_map=label_map,
                source_csv=source_csv,
            )
        finally:
            writer.close()

        combined_view_counts.update(view_counts)
        total_missing_paths.extend(missing_paths)
        summaries[split] = {
            "num_samples": writer.count,
            "num_shards": len(writer.shards),
            "shards": writer.shards,
            "view_counts": dict(sorted(view_counts.items())),
        }

    summary = {
        "dataset": dataset_name,
        "source_csvs": {
            "train": str(train_csv_path),
            "test": str(test_csv_path),
        },
        "num_classes": len(label_map),
        "splits": summaries,
        "view_counts": dict(sorted(combined_view_counts.items())),
        "missing_paths": total_missing_paths,
    }
    write_json(dataset_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ASL datasets packaged as zip archives into WebDataset shards.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where split-specific WebDataset tar shards will be written.",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=1000,
        help="Maximum number of samples stored in each .tar shard.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=["asl_citizen", "wlasl_processed", "wlasl_hub", "wlasl2000_local", "preprocessed_csv", "preprocessed_split_csvs"],
        default=["asl_citizen", "wlasl_processed"],
        help="Subset of supported datasets to process.",
    )
    args = parser.parse_args()

    dataset_sources = {
        "asl_citizen": Path("/home/slime-base/projects/jian/islr/ASL_Citizen.zip"),
        "wlasl_processed": Path("/home/slime-base/projects/jian/islr/wlasl-processed.zip"),
        "wlasl2000_local": Path("/home/slime-base/projects/jian/islr/WLASL2000.zip"),
        "preprocessed_csv": Path("/home/slime-base/projects/jian/islr/data/preprocessed_data_full.csv"),
        "preprocessed_train_csv": Path("/home/slime-base/projects/jian/islr/data/train.csv"),
        "preprocessed_test_csv": Path("/home/slime-base/projects/jian/islr/data/test.csv"),
    }

    summaries = {}
    for dataset_name in args.datasets:
        if dataset_name == "asl_citizen":
            zip_path = dataset_sources[dataset_name]
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing source archive: {zip_path}")
            summaries[dataset_name] = process_asl_citizen(
                zip_path=zip_path,
                output_root=args.output_root,
                max_samples_per_shard=args.max_samples_per_shard,
            )
        elif dataset_name == "wlasl_processed":
            zip_path = dataset_sources[dataset_name]
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing source archive: {zip_path}")
            summaries[dataset_name] = process_wlasl(
                zip_path=zip_path,
                output_root=args.output_root,
                max_samples_per_shard=args.max_samples_per_shard,
            )
        elif dataset_name == "wlasl_hub":
            summaries[dataset_name] = process_wlasl_hub(
                output_root=args.output_root,
                max_samples_per_shard=args.max_samples_per_shard,
            )
        elif dataset_name == "wlasl2000_local":
            zip_path = dataset_sources[dataset_name]
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing source archive: {zip_path}")
            summaries[dataset_name] = process_wlasl2000_local(
                zip_path=zip_path,
                output_root=args.output_root,
                max_samples_per_shard=args.max_samples_per_shard,
            )
        elif dataset_name == "preprocessed_csv":
            csv_path = dataset_sources[dataset_name]
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing source CSV: {csv_path}")
            summaries[dataset_name] = process_preprocessed_csv(
                csv_path=csv_path,
                output_root=args.output_root,
                max_samples_per_shard=args.max_samples_per_shard,
            )
        elif dataset_name == "preprocessed_split_csvs":
            train_csv_path = dataset_sources["preprocessed_train_csv"]
            test_csv_path = dataset_sources["preprocessed_test_csv"]
            if not train_csv_path.exists():
                raise FileNotFoundError(f"Missing source CSV: {train_csv_path}")
            if not test_csv_path.exists():
                raise FileNotFoundError(f"Missing source CSV: {test_csv_path}")
            summaries[dataset_name] = process_preprocessed_split_csvs(
                train_csv_path=train_csv_path,
                test_csv_path=test_csv_path,
                output_root=args.output_root,
                max_samples_per_shard=args.max_samples_per_shard,
            )

    write_json(args.output_root / "conversion_summary.json", summaries)


if __name__ == "__main__":
    main()
