"""Data pipeline for Qwen3 SLT.

Reuses the gesture_pretraining WebDataset shard discovery and decoders, but adds:
  * Uniform downsampling to ``max_frames`` instead of head truncation.
  * Caption normalization (matching t5_slt).
  * A collator that produces variable-length gesture features plus the
    metadata strings consumed by ``Qwen3SLT``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset

from gesture_pretraining.data.dataset import (
    _DATASET_LANGUAGE,
    _FEATURE_KEYS,
    _VAL_ALIASES,
    discover_shards,
)

from t5_slt.data import (
    _GESTURE_DATASET_LANGUAGE,
    get_output_language_name,
    get_sign_language_name,
    normalize_text,
)


def _decode_npz(value: bytes) -> torch.Tensor:
    with np.load(io.BytesIO(value)) as f:
        parts: list[np.ndarray] = []
        expected = None
        for key in _FEATURE_KEYS:
            arr = np.asarray(f[key], dtype=np.float32)
            if expected is None:
                expected = arr.shape[0]
            elif arr.shape[0] != expected:
                raise ValueError(f"temporal dim mismatch for {key}: {arr.shape[0]} vs {expected}")
            parts.append(arr.reshape(arr.shape[0], -1))
    return torch.from_numpy(np.concatenate(parts, axis=1))


def _uniform_downsample(gesture: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Downsample to ``max_frames`` evenly spaced indices when ``T > max_frames``."""
    T = int(gesture.shape[0])
    if T <= max_frames:
        return gesture
    # Even indices from [0, T-1]; ensures last frame is included.
    idx = torch.linspace(0, T - 1, steps=max_frames).round().long()
    return gesture.index_select(0, idx)


def _tag_sample(sample: dict[str, Any]) -> dict[str, Any]:
    url = sample.get("__url__", "")
    parts = Path(url).parts
    ds_name = ""
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in _DATASET_LANGUAGE:
            ds_name = parts[i]
            break
    sample["__dataset__"] = ds_name
    sample["__language__"] = _DATASET_LANGUAGE.get(ds_name, "")
    return sample


def _sample_to_row(sample: dict[str, Any], *, max_frames: int) -> Optional[dict[str, Any]]:
    try:
        npz_bytes = sample.get("npz")
        txt_bytes = sample.get("txt")
        if npz_bytes is None or txt_bytes is None:
            return None
        language = str(sample.get("__language__", "")).strip().lower()
        caption_raw = txt_bytes.decode("utf-8", errors="replace").strip()
        caption = normalize_text(caption_raw, language)
        if not caption:
            return None
        gesture = _decode_npz(npz_bytes).float()
        if gesture.ndim != 2 or gesture.shape[0] <= 0:
            return None
        gesture = _uniform_downsample(gesture, max_frames=max_frames)
        return {
            "gesture": gesture,
            "caption": caption,
            "sample_id": str(sample.get("__key__", "")),
            "dataset_name": str(sample.get("__dataset__", "")),
            "language": language,
            "url": str(sample.get("__url__", "")),
        }
    except Exception:
        return None


def _length_bucket(src: Iterable[dict], *, buffer_size: int, batch_size: int):
    buf: list[dict] = []
    for sample in src:
        buf.append(sample)
        if len(buf) >= buffer_size:
            buf.sort(key=lambda s: int(s["gesture"].shape[0]))
            n_full = len(buf) // batch_size
            for i in range(n_full):
                for s in buf[i * batch_size: (i + 1) * batch_size]:
                    yield s
            buf = buf[n_full * batch_size:]


def build_train_dataset(
    shards: list[str],
    *,
    max_frames: int,
    shuffle_buffer: int = 2000,
    length_bucket_size: int = 0,
    batch_size: int = 0,
) -> wds.DataPipeline:
    """Resampled streaming pipeline for training (infinite stream)."""
    stages: list[Any] = [
        wds.ResampledShards(shards),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.map(_tag_sample),
    ]
    if shuffle_buffer > 0:
        stages.append(wds.shuffle(shuffle_buffer))
    stages.append(wds.map(lambda s: _sample_to_row(s, max_frames=max_frames)))
    stages.append(wds.select(lambda r: r is not None))
    if length_bucket_size > 0 and batch_size > 0:
        stages.append(
            lambda src: _length_bucket(src, buffer_size=length_bucket_size, batch_size=batch_size)
        )
    return wds.DataPipeline(*stages)


def build_eval_dataset(shards: list[str], *, max_frames: int) -> wds.DataPipeline:
    """Deterministic finite traversal for evaluation."""
    return wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.map(_tag_sample),
        wds.map(lambda s: _sample_to_row(s, max_frames=max_frames)),
        wds.select(lambda r: r is not None),
    )


class Qwen3SLTCollator:
    """Pads gesture features and forwards metadata strings; tokenization happens in the model."""

    def __call__(self, rows: list[Optional[dict[str, Any]]]) -> Optional[dict[str, Any]]:
        rows = [r for r in rows if r is not None]
        if not rows:
            return None

        max_t = max(int(r["gesture"].shape[0]) for r in rows)
        feat_dim = int(rows[0]["gesture"].shape[1])

        gestures = torch.zeros(len(rows), max_t, feat_dim, dtype=torch.float32)
        attn_mask = torch.zeros(len(rows), max_t, dtype=torch.long)

        for i, row in enumerate(rows):
            t = int(row["gesture"].shape[0])
            gestures[i, :t] = row["gesture"]
            attn_mask[i, :t] = 1

        languages = [r["language"] for r in rows]
        sign_languages = [get_sign_language_name(l) for l in languages]
        output_languages = [get_output_language_name(l) for l in languages]

        return {
            "gesture": gestures,
            "gesture_attention_mask": attn_mask,
            "captions": [r["caption"] for r in rows],
            "sample_ids": [r["sample_id"] for r in rows],
            "dataset_names": [r["dataset_name"] for r in rows],
            "languages": languages,
            "sign_languages": sign_languages,
            "output_languages": output_languages,
            "urls": [r["url"] for r in rows],
        }


def build_dataloader(
    dataset: IterableDataset,
    *,
    batch_size: int,
    num_workers: int,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=Qwen3SLTCollator(),
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


__all__ = [
    "Qwen3SLTCollator",
    "build_dataloader",
    "build_eval_dataset",
    "build_train_dataset",
    "discover_shards",
    "_GESTURE_DATASET_LANGUAGE",
    "_VAL_ALIASES",
]
