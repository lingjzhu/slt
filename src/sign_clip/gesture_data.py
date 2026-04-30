from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from transformers import PreTrainedTokenizerBase

from t5_slt.data import (
    build_gesture_translation_manifest,
    build_streaming_gesture_translation_dataset,
    discover_gesture_translation_shards,
    load_tar_manifest,
)

from .data import DatasetConfig, load_label_bank
from .text import normalize_sign_text


logger = logging.getLogger(__name__)


_FEATURE_KEYS = (
    "body/global_pose",
    "body/body_pose",
    "body/left_hand_pose",
    "body/right_hand_pose",
    "body/hand_scale",
    "body/head_scale",
    "body/exp",
    "body/shape",
    "flame/eye_pose_params",
    "flame/pose_params",
    "flame/jaw_params",
    "flame/eyelid_params",
    "flame/expression_params",
    "flame/shape_params",
    "camera/pd_cam",
)


DEFAULT_DATASET_CONFIGS = (
    DatasetConfig(name="how2sign_24fps", root=Path(), language="asl", train_split="train", eval_split="val"),
    DatasetConfig(name="openasl_24fps", root=Path(), language="asl", train_split="train", eval_split="val"),
    DatasetConfig(name="dailymoth-70h", root=Path(), language="asl", train_split="train", eval_split="val"),
    DatasetConfig(name="YoutubeASL_processed", root=Path(), language="asl", train_split="train", eval_split="val"),
    DatasetConfig(name="bobsl_24fps", root=Path(), language="bsl", train_split="train", eval_split="val"),
    DatasetConfig(name="bobsl_24fps_manual", root=Path(), language="bsl", train_split="train", eval_split="val"),
    DatasetConfig(name="csl_daily_24fps", root=Path(), language="csl", train_split="train", eval_split="val"),
    DatasetConfig(name="csl_news", root=Path(), language="csl", train_split="train", eval_split="val"),
)


def _load_npz_features(npz_bytes: bytes) -> torch.Tensor:
    with np.load(io.BytesIO(npz_bytes)) as npz_file:
        frames: list[np.ndarray] = []
        expected_length: int | None = None
        for key in _FEATURE_KEYS:
            array = np.asarray(npz_file[key], dtype=np.float32)
            if expected_length is None:
                expected_length = int(array.shape[0])
            elif int(array.shape[0]) != expected_length:
                raise ValueError(f"Inconsistent temporal dimension for {key}: {array.shape[0]} vs {expected_length}")
            frames.append(array.reshape(array.shape[0], -1))
    return torch.from_numpy(np.concatenate(frames, axis=1))


class GestureTarDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_frames: int,
    ) -> None:
        self.examples = load_tar_manifest(manifest_path, max_source_length=max_frames)
        self.lengths = [example.length for example in self.examples]
        self._tar_handles: dict[str, tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _get_tar_handle(self, tar_path: str) -> tarfile.TarFile:
        handle = self._tar_handles.get(tar_path)
        if handle is None:
            handle = tarfile.open(tar_path, "r")
            self._tar_handles[tar_path] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        tar_handle = self._get_tar_handle(example.tar_path)
        member = tar_handle.getmember(example.npz_member)
        with tar_handle.extractfile(member) as handle:
            if handle is None:
                raise FileNotFoundError(f"Missing {example.npz_member} in {example.tar_path}")
            gesture = _load_npz_features(handle.read()).float()

        gesture = gesture[: example.length]
        text = normalize_sign_text(example.caption)
        return {
            "sample_id": example.sample_id,
            "dataset_name": example.dataset_name,
            "language": example.language,
            "target_text": text,
            "gesture": gesture,
            "num_padding_frames": 0,
        }


def _translation_row_to_sign_clip(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "dataset_name": row["dataset_name"],
        "language": row["language"],
        "target_text": normalize_sign_text(row["target_text"]),
        "gesture": row["input_features"],
        "num_padding_frames": 0,
    }


class GestureSignClipCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_text_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, rows: list[Optional[dict[str, Any]]]) -> Optional[dict[str, Any]]:
        rows = [row for row in rows if row is not None]
        if not rows:
            return None

        max_t = max(int(row["gesture"].shape[0]) for row in rows)
        feat_dim = int(rows[0]["gesture"].shape[1])
        gestures = torch.zeros(len(rows), max_t, feat_dim, dtype=torch.float32)
        attention_mask = torch.zeros(len(rows), max_t, dtype=torch.long)
        num_padding_frames = torch.zeros(len(rows), dtype=torch.long)

        for i, row in enumerate(rows):
            gesture = row["gesture"]
            t = int(gesture.shape[0])
            gestures[i, :t] = gesture
            attention_mask[i, :t] = 1
            num_padding_frames[i] = max_t - t

        tokenized = self.tokenizer(
            [row["target_text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        batch = {
            "gesture": gestures,
            "gesture_attention_mask": attention_mask,
            "num_padding_frames": num_padding_frames,
            "target_texts": [row["target_text"] for row in rows],
            "sample_ids": [row["sample_id"] for row in rows],
            "dataset_names": [row["dataset_name"] for row in rows],
            "languages": [row["language"] for row in rows],
        }
        for key, value in tokenized.items():
            batch[f"text_{key}"] = value
        return batch


def prepare_gesture_manifests(
    *,
    manifest_dir: str | Path,
    data_root: str | Path,
    dataset_configs: list[DatasetConfig],
) -> dict[str, Path]:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    datasets = {config.name for config in dataset_configs}
    languages = {config.language for config in dataset_configs}
    paths = {
        "train": manifest_dir / "train_gesture.tsv",
        "val": manifest_dir / "val_gesture.tsv",
        "test": manifest_dir / "test_gesture.tsv",
    }
    for split, path in paths.items():
        rebuild = not path.exists()
        if not rebuild:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    header = handle.readline().rstrip("\n").split("\t")
                    first = handle.readline().rstrip("\n").split("\t")
                tar_idx = header.index("tar_path")
                first_tar = Path(first[tar_idx])
                rebuild = not first_tar.is_relative_to(Path(data_root))
            except (IndexError, OSError, ValueError):
                rebuild = True

        if rebuild:
            build_gesture_translation_manifest(
                split=split,
                manifest_path=path,
                data_root=data_root,
                datasets=datasets,
                languages=languages,
            )
    return paths


def build_train_dataset(
    dataset_configs: list[DatasetConfig],
    *,
    manifest_dir: str | Path,
    data_root: str | Path,
    max_frames: int,
) -> Any:
    datasets = {config.name for config in dataset_configs}
    languages = {config.language for config in dataset_configs}
    shards = discover_gesture_translation_shards(
        data_root,
        "train",
        datasets=datasets,
        languages=languages,
    )
    logger.info("streaming %d gesture train shards from %s", len(shards), data_root)
    dataset = build_streaming_gesture_translation_dataset(
        shards,
        split="train",
        prompt_template="",
        max_source_length=max_frames,
    )
    import webdataset as wds

    dataset.append(wds.map(_translation_row_to_sign_clip))
    return dataset


def build_eval_dataset(
    config: DatasetConfig,
    *,
    manifest_dir: str | Path,
    data_root: str | Path,
    max_frames: int,
) -> GestureTarDataset:
    manifests = prepare_gesture_manifests(
        manifest_dir=manifest_dir,
        data_root=data_root,
        dataset_configs=[config],
    )
    split = "val" if config.eval_split == "val" else "test"
    return GestureTarDataset(manifests[split], max_frames=max_frames)


def build_dataloader(
    dataset: Any,
    *,
    collate_fn,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    sampler: Optional[DistributedSampler] = None
    is_iterable = isinstance(dataset, IterableDataset) or not hasattr(dataset, "__len__")
    if (
        not is_iterable
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
        shuffle = False

    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if not is_iterable:
        kwargs["shuffle"] = shuffle
        kwargs["sampler"] = sampler
    loader = DataLoader(dataset, **kwargs)
    return loader, sampler


__all__ = [
    "DEFAULT_DATASET_CONFIGS",
    "GestureSignClipCollator",
    "GestureTarDataset",
    "build_dataloader",
    "build_eval_dataset",
    "build_train_dataset",
    "load_label_bank",
    "prepare_gesture_manifests",
]
