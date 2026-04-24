from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .data import DatasetConfig, load_label_bank
from .text import normalize_sign_text


DEFAULT_LATENT_WEBDS_ROOT = Path(
    os.environ.get(
        "SIGN_CLIP_LATENT_WEBDS_ROOT",
        "/home/slimelab/Projects/slt/islr/webdataset_224_leanvae_latents",
    )
)


DEFAULT_LATENT_DATASET_CONFIGS = (
    DatasetConfig(
        name="wlasl",
        root=DEFAULT_LATENT_WEBDS_ROOT / "wlasl",
        language="asl",
        train_split="train",
        eval_split="val",
    ),
    DatasetConfig(
        name="asl_citizen",
        root=DEFAULT_LATENT_WEBDS_ROOT / "asl_citizen",
        language="asl",
        train_split="train",
        eval_split="val",
    ),
    DatasetConfig(
        name="csl_large",
        root=DEFAULT_LATENT_WEBDS_ROOT / "csl_large",
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


def _load_pt_tensor(payload: bytes) -> torch.Tensor:
    tensor = torch.load(io.BytesIO(payload), map_location="cpu")
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor payload, got {type(tensor)!r}")
    if tensor.ndim != 4:
        raise ValueError(f"Expected latent tensor shape [T, C, H, W], got {tuple(tensor.shape)}")
    return tensor.float()


def _cap_temporal_axis(latent: torch.Tensor, *, max_num_frames: int) -> torch.Tensor:
    src_frames = int(latent.shape[0])
    if src_frames > max_num_frames:
        return latent[:max_num_frames]
    return latent


def _extract_text(metadata: dict[str, Any], txt_payload: bytes | str | None) -> str:
    if isinstance(txt_payload, bytes):
        text = txt_payload.decode("utf-8", errors="replace").strip()
        if text:
            return normalize_sign_text(text)
    elif isinstance(txt_payload, str) and txt_payload.strip():
        return normalize_sign_text(txt_payload)

    for key in ("caption", "gloss", "transcription", "text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_sign_text(value)
    return ""


def _process_latent_sample(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    language: str,
    training: bool,
    num_frames: int,
) -> Optional[dict[str, Any]]:
    try:
        metadata_raw = sample.get("json")
        latent_raw = sample.get("pt")
        txt_raw = sample.get("txt")
        if metadata_raw is None or latent_raw is None:
            return None

        metadata = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw)
        text = _extract_text(metadata, txt_raw)
        if not text:
            return None

        latent = _load_pt_tensor(latent_raw)
        latent = _cap_temporal_axis(latent, max_num_frames=num_frames)
        return {
            "sample_id": metadata.get("key", sample.get("__key__", "")),
            "dataset_name": dataset_name,
            "language": language,
            "target_text": text,
            "latents": latent,
            "num_padding_frames": 0,
            "input_num_frames": int(metadata.get("vae_latent", {}).get("input_num_frames", 0)),
            "latent_num_frames": int(metadata.get("vae_latent", {}).get("latent_num_frames", latent.shape[0])),
        }
    except Exception:
        return None


def _build_component_dataset(
    config: DatasetConfig,
    *,
    split: str,
    training: bool,
    num_frames: int,
    shard_shuffle: bool,
    sample_shuffle: int,
    resampled: bool,
    distributed_split: bool = True,
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
        dataset.select(lambda sample: "pt" in sample and "json" in sample)
        .map(
            lambda sample: _process_latent_sample(
                sample,
                dataset_name=config.name,
                language=config.language,
                training=training,
                num_frames=num_frames,
            )
        )
        .select(lambda sample: sample is not None)
    )


class LatentSignClipCollator:
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

        max_t = max(int(row["latents"].shape[0]) for row in rows)
        padded_latents = []
        pad_counts = []
        for row in rows:
            latent = row["latents"]
            t = int(latent.shape[0])
            if t < max_t:
                pad = latent[-1:].expand(max_t - t, -1, -1, -1)
                latent = torch.cat([latent, pad], dim=0)
            padded_latents.append(latent)
            pad_counts.append(max_t - t)
        latents = torch.stack(padded_latents, dim=0)
        num_padding_frames = torch.tensor(pad_counts, dtype=torch.long)
        tokenized = self.tokenizer(
            [row["target_text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        batch = {
            "latents": latents,
            "num_padding_frames": num_padding_frames,
            "target_texts": [row["target_text"] for row in rows],
            "sample_ids": [row["sample_id"] for row in rows],
            "dataset_names": [row["dataset_name"] for row in rows],
            "languages": [row["language"] for row in rows],
            "input_num_frames": torch.tensor([row["input_num_frames"] for row in rows], dtype=torch.long),
            "latent_num_frames": torch.tensor([row["latent_num_frames"] for row in rows], dtype=torch.long),
        }
        for key, value in tokenized.items():
            batch[f"text_{key}"] = value
        return batch


def build_train_dataset(
    dataset_configs: list[DatasetConfig],
    *,
    num_frames: int,
    sample_shuffle: int = 256,
    equal_dataset_mix: bool = True,
) -> Any:
    components = [
        _build_component_dataset(
            config,
            split=config.train_split,
            training=True,
            num_frames=num_frames,
            shard_shuffle=True,
            sample_shuffle=sample_shuffle,
            resampled=equal_dataset_mix,
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
) -> Any:
    return _build_component_dataset(
        config,
        split=config.eval_split,
        training=False,
        num_frames=num_frames,
        shard_shuffle=False,
        sample_shuffle=0,
        resampled=False,
        distributed_split=False,
    )


def build_dataloader(
    dataset: Any,
    *,
    collate_fn,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "DEFAULT_LATENT_DATASET_CONFIGS",
    "DEFAULT_LATENT_WEBDS_ROOT",
    "LatentSignClipCollator",
    "build_dataloader",
    "build_eval_dataset",
    "build_train_dataset",
    "load_label_bank",
]
