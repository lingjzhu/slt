"""Raw-video dataset for stage-2 V-JEPA-style training.

Reads a t5_slt-format detailed manifest (video_path + caption + language + ...),
decodes mp4 frames, and returns a `[C, T, H, W]` clip ready for SignHiera.
The EMA teacher consumes the same clip to produce pooled per-frame targets
online, and DDP synchronizes the student SignHiera weights.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

_MAE_SRC = Path(__file__).resolve().parents[1]
if str(_MAE_SRC) not in sys.path:
    sys.path.insert(0, str(_MAE_SRC))

from mae_pretraining.utils.video import (  # noqa: E402
    get_num_padding_frames,
    get_start_end_idx,
    temporal_sampling,
    tensor_normalize,
    uniform_crop,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RawVideoExample:
    sample_id: str
    video_path: str
    caption: str
    language: str


def _resolve_video_path(manifest_path: Path, raw: str, video_root: Optional[Path] = None) -> str:
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if video_root is not None:
        return str((video_root / p).resolve())
    # t5_slt convention: manifest lives at <split_root>/manifests/<file>.tsv,
    # videos live at <split_root>/videos/...
    split_root = manifest_path.parent.parent
    return str((split_root / p).resolve())


def load_raw_manifest(
    manifest_path: str | Path,
    *,
    video_root: Optional[str | Path] = None,
) -> list[RawVideoExample]:
    manifest_path = Path(manifest_path)
    root = Path(video_root) if video_root else None
    rows: list[RawVideoExample] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                RawVideoExample(
                    sample_id=Path(row["video_path"]).stem,
                    video_path=_resolve_video_path(manifest_path, row["video_path"], root),
                    caption=row.get("caption", "").strip(),
                    language=row.get("language", "").strip().lower(),
                )
            )
    if not rows:
        raise ValueError(f"No rows in {manifest_path}")
    return rows


def _decode_pyav(path: str) -> tuple[torch.Tensor, float]:
    import torchvision
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(path, "video")
    meta = reader.get_metadata()
    fps = float(meta["video"]["fps"][0])
    frames = []
    for f in reader:
        frames.append(f["data"])
    frames_t = torch.stack(frames)  # [T, C, H, W]
    # return [T, H, W, C] uint8 to match the temporal_sampling / tensor_normalize conventions
    return frames_t.permute(0, 2, 3, 1).contiguous(), fps


class RawVideoDataset(Dataset):
    def __init__(
        self,
        examples: list[RawVideoExample],
        *,
        num_frames: int = 128,
        sampling_rate: int = 1,
        target_fps: float = 8.0,
        crop_size: int = 224,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.examples = examples
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.target_fps = target_fps
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Optional[dict[str, Any]]:
        ex = self.examples[index]
        try:
            frames, fps = _decode_pyav(ex.video_path)  # [T, H, W, C] uint8
        except Exception as exc:
            print(f"skip bad video {ex.video_path}: {exc}", flush=True)
            return None

        clip_sz = self.sampling_rate * self.num_frames / self.target_fps * fps
        start, end = get_start_end_idx(frames.shape[0], clip_sz, -1, 1, use_offset=True)
        clip, idx = temporal_sampling(frames, start, end, self.num_frames, return_index=True)
        num_padding_frames = get_num_padding_frames(
            idx, self.num_frames, self.sampling_rate, fps, self.target_fps
        )
        clip = tensor_normalize(clip, self.mean, self.std)  # [T, H, W, C] float
        clip = clip.permute(3, 0, 1, 2)  # [C, T, H, W]
        clip = uniform_crop(clip, self.crop_size, spatial_idx=1)  # center crop

        return {
            "sample_id": ex.sample_id,
            "video": clip,  # [C, T, H, W]
            "num_padding_frames": int(num_padding_frames),
            "prompt_text": "",
            "target_text": ex.caption,
            "language": ex.language,
        }


class RawVideoCollator:
    """Collator for raw-video stage-2 training.

    Reuses the text half of DiscreteDiffusionCollator (prompt+target tokenization,
    prefix guard, paired masks) and adds a visual half that stacks clips and
    samples per-frame visual masks matching the student's pooled output length.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_text_length: int = 128,
        prompt_template: str = "translate into {language}:",
        language_map: Optional[dict[str, str]] = None,
        pooled_frames: int,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ) -> None:
        # Import lazily to avoid a circular import at module load.
        from .data import DiscreteDiffusionCollator, sample_masks

        self._sample_masks = sample_masks
        self.text = DiscreteDiffusionCollator(
            tokenizer, max_text_length=max_text_length, min_ratio=min_ratio, max_ratio=max_ratio
        )
        self.prompt_template = prompt_template
        self.language_map = language_map or {
            "asl": "English", "bsl": "English", "csl": "Chinese",
            "en": "English", "zh": "Chinese",
        }
        self.pooled_frames = pooled_frames
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.input_frames: Optional[int] = None  # set on first batch

    def _language_name(self, code: str) -> str:
        c = (code or "").strip().lower()
        return self.language_map.get(c, c.capitalize() or "English")

    def __call__(self, rows: list[Optional[dict]]) -> Optional[dict[str, Any]]:
        rows = [r for r in rows if r is not None]
        if not rows:
            return None

        for r in rows:
            r["prompt_text"] = self.prompt_template.format(language=self._language_name(r["language"]))
            # provide dummy tensors for the text-half collator
            r["visual_features"] = torch.zeros(self.pooled_frames, 1)
            r["visual_targets"] = torch.zeros(self.pooled_frames, 1)

        text_batch = self.text(rows)
        # drop the dummy visual tensors the text collator produced
        for k in ("visual_features", "visual_targets"):
            text_batch.pop(k, None)

        videos = torch.stack([r["video"] for r in rows])  # [B, C, T, H, W]

        B = videos.shape[0]
        T_out = self.pooled_frames
        T_in = videos.shape[2]
        # Convert input-frame padding to pooled-frame padding. Round DOWN to
        # avoid marking valid pooled positions as padding.
        vis_attn = torch.ones(B, T_out, dtype=torch.long)
        pooled_pad = torch.zeros(B, dtype=torch.long)
        for i, r in enumerate(rows):
            pad_in = int(r.get("num_padding_frames", 0))
            pad_out = (pad_in * T_out) // T_in if T_in > 0 else 0
            pad_out = max(0, min(pad_out, T_out))
            pooled_pad[i] = pad_out
            if pad_out > 0:
                vis_attn[i, T_out - pad_out :] = 0

        ratios = text_batch["corruption_ratios"]
        vis_a, vis_b = self._sample_masks(valid_mask=vis_attn.to(torch.uint8), ratios=ratios)

        text_batch["visual_raw"] = videos
        text_batch["visual_attention_mask"] = vis_attn
        text_batch["visual_mask_a"] = vis_a
        text_batch["visual_mask_b"] = vis_b
        text_batch["visual_num_padding_frames_in"] = torch.tensor(
            [int(r.get("num_padding_frames", 0)) for r in rows], dtype=torch.long
        )
        text_batch["visual_num_padding_frames_pooled"] = pooled_pad
        return text_batch
