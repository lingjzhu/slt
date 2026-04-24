from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import av
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizerBase

try:
    import decord as _decord
    _decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

from mae_pretraining.utils.video import (
    get_num_padding_frames,
    get_start_end_idx,
    tensor_normalize,
    uniform_crop,
    random_crop,
    horizontal_flip,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


LANGUAGE_PROMPTS: dict[str, str] = {
    "asl": "Translate American Sign Language to English: ",
    "bsl": "Translate British Sign Language to English: ",
    "csl": "Translate Chinese Sign Language to Chinese: ",
}


def build_prompted_text(language: str, text: str) -> str:
    prefix = LANGUAGE_PROMPTS.get(language.lower(), f"Translate {language} sign language: ")
    return prefix + text


@dataclass(frozen=True)
class PairedDataConfig:
    base_data_dir: Path
    dataset_name: str = "all_train_plain_v3"
    manifest_candidates: tuple[str, ...] = (
        "final_train_le16.tsv",
        "paired_manifest.tsv",
        "train.tsv",
    )
    min_duration: float = 0.0
    max_duration: float = math.inf
    allowed_languages: Optional[tuple[str, ...]] = None

    @property
    def data_dir(self) -> Path:
        return self.base_data_dir / self.dataset_name

    def resolve_manifest(self) -> Path:
        manifest_dir = self.data_dir / "manifests"
        for name in self.manifest_candidates:
            candidate = manifest_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No manifest found under {manifest_dir}")


class PairedSignDataset(Dataset):
    def __init__(
        self,
        config: PairedDataConfig,
        *,
        training: bool,
        num_frames: int,
        sampling_rate: int,
        target_fps: float,
        crop_size: int,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        random_spatial_crop: bool = True,
        random_horizontal_flip: bool = False,
    ) -> None:
        self.config = config
        self.training = training
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.target_fps = float(target_fps)
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.random_spatial_crop = random_spatial_crop and training
        self.random_horizontal_flip = random_horizontal_flip and training

        manifest_path = config.resolve_manifest()
        videos_root = config.data_dir / "videos"
        allowed = (
            tuple(lang.lower() for lang in config.allowed_languages)
            if config.allowed_languages is not None
            else None
        )

        self.samples: list[tuple[str, str, str]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                video_name, duration_s, _dataset, language, text = (
                    fields[0], fields[1], fields[2], fields[3], "\t".join(fields[4:])
                )
                try:
                    duration = float(duration_s)
                except ValueError:
                    continue
                if duration < config.min_duration or duration > config.max_duration:
                    continue
                if allowed is not None and language.lower() not in allowed:
                    continue
                if not text.strip():
                    continue
                if os.path.isabs(video_name):
                    video_path = video_name
                else:
                    prefix = video_name[:5]
                    video_path = str(videos_root / prefix / video_name)
                self.samples.append((video_path, language.lower(), text))

        if not self.samples:
            raise RuntimeError(f"No samples loaded from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _plan_indices(self, total_frames: int, fps: float) -> tuple[torch.Tensor, int]:
        """Pick `num_frames` integer indices at `target_fps` (linear sample within a
        random / centered clip window), mirroring temporal_sampling but returning
        only indices so the decoder can seek to them directly."""
        clip_sz = self.sampling_rate * self.num_frames / self.target_fps * fps
        clip_idx = -1 if self.training else 0
        start, end = get_start_end_idx(total_frames, clip_sz, clip_idx, 1, use_offset=True)
        idx = torch.linspace(float(start), float(end), self.num_frames)
        idx = torch.clamp(idx, 0, max(total_frames - 1, 0)).long()
        num_pad = get_num_padding_frames(idx, self.num_frames, self.sampling_rate, fps, self.target_fps)
        return idx, int(num_pad)

    def _decode_decord(self, path: str) -> tuple[torch.Tensor, int]:
        vr = _decord.VideoReader(path, num_threads=1)
        fps = vr.get_avg_fps() or 25.0
        total = len(vr)
        idx, num_pad = self._plan_indices(total, float(fps))
        frames = vr.get_batch(idx.tolist())  # [T, H, W, C] uint8
        return frames, num_pad

    def _decode_pyav(self, path: str) -> tuple[torch.Tensor, int]:
        frames = []
        with av.open(path, mode="r") as container:
            stream = container.streams.video[0]
            if stream.average_rate is not None:
                fps = float(stream.average_rate)
            elif stream.base_rate is not None:
                fps = float(stream.base_rate)
            else:
                fps = float(stream.guessed_rate or 25.0)
            stream.thread_type = "AUTO"
            total = stream.frames or 0
            if total <= 0:
                for frame in container.decode(stream):
                    frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
                total = len(frames)
                idx, num_pad = self._plan_indices(total, fps)
                stacked = torch.stack(frames, dim=0).index_select(0, idx)
                return stacked, num_pad
            idx, num_pad = self._plan_indices(total, fps)
            wanted = set(int(i) for i in idx.tolist())
            max_idx = max(wanted)
            collected: dict[int, torch.Tensor] = {}
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    collected[i] = torch.from_numpy(frame.to_rgb().to_ndarray())
                    if len(collected) == len(wanted) or i >= max_idx:
                        break
            ordered = [collected[int(i)] for i in idx.tolist() if int(i) in collected]
            if not ordered:
                raise ValueError("decoded zero frames")
            return torch.stack(ordered, dim=0), num_pad

    def _sample(self, path: str) -> tuple[torch.Tensor, int]:
        if _HAS_DECORD:
            try:
                return self._decode_decord(path)
            except Exception:
                pass
        return self._decode_pyav(path)

    def _postprocess(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [T, H, W, C] uint8 -> [C, T, H, W] normalized float
        clip = tensor_normalize(clip, self.mean, self.std).permute(3, 0, 1, 2).contiguous()
        if self.random_spatial_crop:
            clip = random_crop(clip, self.crop_size)
        else:
            clip = uniform_crop(clip, self.crop_size, spatial_idx=1)
        if self.random_horizontal_flip:
            clip = horizontal_flip(0.5, clip)
        return clip

    def __getitem__(self, index: int) -> Optional[dict[str, Any]]:
        video_path, language, text = self.samples[index]
        try:
            frames, num_pad = self._sample(video_path)
            clip = self._postprocess(frames)
        except Exception:
            # fall back to a different random sample rather than returning None mid-DDP
            return self.__getitem__(random.randrange(len(self.samples)))
        return {
            "video": clip,
            "num_padding_frames": num_pad,
            "language": language,
            "text": text,
            "prompted_text": build_prompted_text(language, text),
        }


class PairedSignCollator:
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
            [row["prompted_text"] for row in rows],
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
            "target_texts": [row["prompted_text"] for row in rows],
            "raw_texts": [row["text"] for row in rows],
            "languages": [row["language"] for row in rows],
        }
        for key, value in tokenized.items():
            batch[f"text_{key}"] = value
        return batch


def build_paired_dataloader(
    dataset: PairedSignDataset,
    *,
    collate_fn: PairedSignCollator,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    ddp_enabled: bool,
    rank: int,
    world_size: int,
    seed: int,
    shuffle: bool,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    sampler: Optional[DistributedSampler] = None
    if ddp_enabled:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return loader, sampler
