from __future__ import annotations

import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import av
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mae_pretraining.utils.video import (
    get_num_padding_frames,
    get_start_end_idx,
    temporal_sampling,
    tensor_normalize,
    uniform_crop,
)

from .text import extract_text_label, normalize_sign_text


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


DEFAULT_WEBDS_ROOT = Path(
    os.environ.get("SIGN_CLIP_WEBDS_ROOT", "/home/slimelab/Projects/slt/islr/webdataset_224")
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    language: str
    train_split: str = "train"
    eval_split: str = "val"
    keep_native_fps: bool = False


DEFAULT_DATASET_CONFIGS = (
    DatasetConfig(
        name="wlasl",
        root=DEFAULT_WEBDS_ROOT / "wlasl",
        language="asl",
        train_split="train",
        eval_split="val",
    ),
    DatasetConfig(
        name="asl_citizen",
        root=DEFAULT_WEBDS_ROOT / "asl_citizen",
        language="asl",
        train_split="train",
        eval_split="val",
    ),
    DatasetConfig(
        name="csl_large",
        root=DEFAULT_WEBDS_ROOT / "csl_large",
        language="csl",
        train_split="train",
        eval_split="test",
        keep_native_fps=True,
    ),
)


def _dataset_urls(root: Path, split: str) -> list[str]:
    split_dir = root / split
    urls = sorted(str(path) for path in split_dir.glob("*.tar"))
    if not urls:
        raise FileNotFoundError(f"No shards found in {split_dir}")
    return urls


def _decode_video_bytes(payload: bytes) -> tuple[torch.Tensor, float]:
    frames = []
    with av.open(io.BytesIO(payload), mode="r", format="mp4") as container:
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        elif stream.base_rate is not None:
            fps = float(stream.base_rate)
        else:
            fps = float(stream.guessed_rate or 25.0)
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
    if not frames:
        raise ValueError("decoded zero frames")
    return torch.stack(frames, dim=0), fps


def _process_sample(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    language: str,
    keep_native_fps: bool,
    training: bool,
    num_frames: int,
    sampling_rate: int,
    target_fps: float,
    crop_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    no_resample: bool = False,
    output_dtype: str = "float32",
) -> Optional[dict[str, Any]]:
    try:
        metadata_raw = sample.get("json")
        video_raw = sample.get("mp4")
        if metadata_raw is None or video_raw is None:
            return None

        metadata = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw)
        text = extract_text_label(metadata)
        if not text:
            return None

        frames, fps = _decode_video_bytes(video_raw)
        if no_resample:
            t_src = frames.shape[0]
            if t_src > num_frames:
                idx = torch.linspace(0, t_src - 1, num_frames).round().long()
                clip = frames.index_select(0, idx)
                num_padding_frames = 0
            else:
                pad = torch.zeros(
                    (num_frames - t_src, *frames.shape[1:]), dtype=frames.dtype
                )
                clip = torch.cat([frames, pad], dim=0)
                num_padding_frames = num_frames - t_src
        else:
            effective_target_fps = fps if keep_native_fps else target_fps
            clip_size = sampling_rate * num_frames / effective_target_fps * fps
            clip_idx = -1 if training else 0
            start, end = get_start_end_idx(frames.shape[0], clip_size, clip_idx, 1, use_offset=True)
            clip, idx = temporal_sampling(frames, start, end, num_frames, return_index=True)
            num_padding_frames = get_num_padding_frames(
                idx, num_frames, sampling_rate, fps, effective_target_fps
            )

        # Skip CPU normalization if the caller wants uint8 — normalization (uint8
        # -> bf16, /255, -mean, /std) is then deferred to the GPU inside the
        # model, which is ~100x faster and halves the H2D transfer (uint8 vs
        # float32). Keep permute + crop on uint8 (both are view/stride ops).
        if output_dtype == "uint8":
            clip = clip.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W] uint8
        else:
            clip = tensor_normalize(clip, mean, std).permute(3, 0, 1, 2).contiguous()
        clip = uniform_crop(clip, crop_size, spatial_idx=1)

        return {
            "sample_id": metadata.get("key", sample.get("__key__", "")),
            "dataset_name": dataset_name,
            "language": language,
            "target_text": text,
            "video": clip,
            "num_padding_frames": int(num_padding_frames),
        }
    except Exception:
        return None


def _build_component_dataset(
    config: DatasetConfig,
    *,
    split: str,
    training: bool,
    num_frames: int,
    sampling_rate: int,
    target_fps: float,
    crop_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    shard_shuffle: bool,
    sample_shuffle: int,
    resampled: bool,
    no_resample: bool = False,
    distributed_split: bool = True,
    output_dtype: str = "float32",
) -> wds.WebDataset:
    dataset = wds.WebDataset(
        _dataset_urls(config.root, split),
        shardshuffle=shard_shuffle,
        resampled=resampled,
        handler=wds.warn_and_continue,
        nodesplitter=wds.split_by_node if distributed_split else None,
        workersplitter=wds.split_by_worker if distributed_split else None,
        empty_check=False,
    )
    if sample_shuffle > 0:
        dataset = dataset.shuffle(sample_shuffle)
    return (
        dataset.select(lambda sample: "mp4" in sample and "json" in sample)
        .map(
            lambda sample: _process_sample(
                sample,
                dataset_name=config.name,
                language=config.language,
                keep_native_fps=config.keep_native_fps,
                training=training,
                num_frames=num_frames,
                sampling_rate=sampling_rate,
                target_fps=target_fps,
                crop_size=crop_size,
                mean=mean,
                std=std,
                no_resample=no_resample,
                output_dtype=output_dtype,
            )
        )
        .select(lambda sample: sample is not None)
    )


class SignClipCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_text_length: int,
        pooled_frames: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pooled_frames = pooled_frames

    def __call__(self, rows: list[Optional[dict[str, Any]]]) -> Optional[dict[str, Any]]:
        rows = [row for row in rows if row is not None]
        if not rows:
            return None

        videos = torch.stack([row["video"] for row in rows], dim=0)
        tokenized = self.tokenizer(
            [row["target_text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        batch_size = videos.shape[0]
        pooled_attention_mask = torch.ones(batch_size, self.pooled_frames, dtype=torch.long)
        input_frames = videos.shape[2]
        num_padding_frames = torch.zeros(batch_size, dtype=torch.long)
        for i, row in enumerate(rows):
            pad_in = int(row.get("num_padding_frames", 0))
            num_padding_frames[i] = pad_in
            pad_out = (pad_in * self.pooled_frames) // max(input_frames, 1)
            if pad_out > 0:
                pooled_attention_mask[i, self.pooled_frames - pad_out :] = 0

        batch = {
            "video": videos,
            "video_attention_mask": pooled_attention_mask,
            "video_num_padding_frames": num_padding_frames,
            "target_texts": [row["target_text"] for row in rows],
            "sample_ids": [row["sample_id"] for row in rows],
            "dataset_names": [row["dataset_name"] for row in rows],
            "languages": [row["language"] for row in rows],
        }
        for key, value in tokenized.items():
            batch[f"text_{key}"] = value
        return batch


def build_train_dataset(
    dataset_configs: list[DatasetConfig],
    *,
    num_frames: int,
    sampling_rate: int,
    target_fps: float,
    crop_size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    sample_shuffle: int = 256,
    equal_dataset_mix: bool = True,
    no_resample: bool = False,
    output_dtype: str = "float32",
) -> Any:
    components = [
        _build_component_dataset(
            config,
            split=config.train_split,
            training=True,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            target_fps=target_fps,
            crop_size=crop_size,
            mean=mean,
            std=std,
            shard_shuffle=True,
            sample_shuffle=sample_shuffle,
            resampled=equal_dataset_mix,
            no_resample=no_resample,
            output_dtype=output_dtype,
        )
        for config in dataset_configs
    ]
    if equal_dataset_mix and len(components) > 1:
        probs = [1.0 / len(components)] * len(components)
        return wds.RandomMix(components, probs=probs)
    if len(components) == 1:
        return components[0]
    return wds.RandomMix(components)


def build_eval_dataset(
    config: DatasetConfig,
    *,
    num_frames: int,
    sampling_rate: int,
    target_fps: float,
    crop_size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    no_resample: bool = False,
    output_dtype: str = "float32",
) -> Any:
    return _build_component_dataset(
        config,
        split=config.eval_split,
        training=False,
        num_frames=num_frames,
        sampling_rate=sampling_rate,
        target_fps=target_fps,
        crop_size=crop_size,
        mean=mean,
        std=std,
        shard_shuffle=False,
        sample_shuffle=0,
        resampled=False,
        no_resample=no_resample,
        distributed_split=False,
        output_dtype=output_dtype,
    )


def build_dataloader(
    dataset: Any,
    *,
    collate_fn,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def load_label_bank(config: DatasetConfig) -> list[str]:
    label_map_path = config.root / "label_map.json"
    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    labels = payload["labels"]

    ordered: list[str] = []
    if config.name == "csl_large":
        entries = sorted(labels.items(), key=lambda item: int(item[1]["index"]))
        ordered = [normalize_sign_text(entry["transcription"]) for _, entry in entries]
    else:
        ordered = [
            text
            for text, _index in sorted(labels.items(), key=lambda item: int(item[1]))
        ]
        ordered = [normalize_sign_text(text) for text in ordered]

    deduped: list[str] = []
    seen: set[str] = set()
    for text in ordered:
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)
    if not deduped:
        raise ValueError(f"No labels loaded from {label_map_path}")
    return deduped
