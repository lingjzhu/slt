from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class DiffusionExample:
    sample_id: str
    feature_path: str
    teacher_feature_path: Optional[str]
    prompt_text: str
    target_text: str
    language: str


def sample_masks(
    *,
    valid_mask: torch.Tensor,
    ratios: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a base mask and its complement over valid positions.

    `valid_mask` is `[B, L]` of 1 where position is maskable. `ratios` is `[B]`.
    Returns `(base_mask, inverse_mask)` both `[B, L]` uint8, partitioning the
    valid positions exactly.
    """
    B, L = valid_mask.shape
    probs = ratios.view(B, 1).expand(B, L)
    u = torch.rand((B, L), generator=generator, device=valid_mask.device)
    base = (u < probs) & valid_mask.bool()
    inverse = (~base) & valid_mask.bool()
    return base.to(torch.uint8), inverse.to(torch.uint8)


class DiscreteDiffusionDataset(Dataset):
    """Dataset returning pooled student features, teacher targets, and text.

    Each row is a dict of python / tensor objects; collation happens separately.
    """

    def __init__(self, examples: list[DiffusionExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ex = self.examples[index]
        feats = _load_pooled(ex.feature_path)
        if ex.teacher_feature_path and Path(ex.teacher_feature_path).exists():
            teacher = _load_pooled(ex.teacher_feature_path)
        else:
            teacher = feats.clone()
        t = min(feats.size(0), teacher.size(0))
        return {
            "sample_id": ex.sample_id,
            "visual_features": feats[:t],
            "visual_targets": teacher[:t],
            "prompt_text": ex.prompt_text,
            "target_text": ex.target_text,
            "language": ex.language,
            "length": int(t),
        }


def _load_pooled(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("features", "hidden_states", "x"):
            if k in obj and torch.is_tensor(obj[k]):
                obj = obj[k]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected tensor at {path}")
    t = obj.float()
    if t.ndim != 2:
        raise ValueError(f"Expected [time, dim] features at {path}, got {tuple(t.shape)}")
    return t


class DiscreteDiffusionCollator:
    """Collator producing batched tensors plus paired mask/inverse-mask."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_text_length: int = 128,
        prompt_separator: str = " ",
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.prompt_separator = prompt_separator
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must define a mask token for discrete diffusion.")
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        # tokens that should never be masked (pad + all special tokens)
        specials = set(tokenizer.all_special_ids or [])
        specials.discard(self.mask_token_id)
        self._non_maskable_ids = torch.tensor(sorted(specials), dtype=torch.long)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        B = len(rows)
        max_T = max(r["visual_features"].size(0) for r in rows)
        D_s = rows[0]["visual_features"].size(1)
        D_t = rows[0]["visual_targets"].size(1)

        visual_features = torch.zeros(B, max_T, D_s)
        visual_targets = torch.zeros(B, max_T, D_t)
        visual_attention_mask = torch.zeros(B, max_T, dtype=torch.long)
        prompt_lengths = torch.zeros(B, dtype=torch.long)

        for i, r in enumerate(rows):
            t = r["visual_features"].size(0)
            visual_features[i, :t] = r["visual_features"]
            visual_targets[i, :t] = r["visual_targets"]
            visual_attention_mask[i, :t] = 1

        # Tokenize prompt and target separately so we know exactly which positions
        # belong to the (non-maskable) prompt prefix. Build sequences as:
        #   [BOS?] + prompt_ids + target_ids + [EOS?]
        # using the tokenizer's own special-token builder to stay consistent.
        prompt_ids_list: list[list[int]] = self.tokenizer(
            [r["prompt_text"] for r in rows],
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
        )["input_ids"]
        target_ids_list: list[list[int]] = self.tokenizer(
            [r["target_text"] for r in rows],
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
        )["input_ids"]

        # Reserve budget for specials. Use build_inputs_with_special_tokens on a
        # tiny sentinel to count how many specials the tokenizer adds per side.
        probe = self.tokenizer.build_inputs_with_special_tokens([0])
        num_leading = probe.index(0)
        num_trailing = len(probe) - num_leading - 1
        budget = self.max_text_length - num_leading - num_trailing

        full_ids: list[list[int]] = []
        prompt_offsets: list[int] = []  # where prompt starts within the full sequence
        prompt_lens_list: list[int] = []
        for p_ids, t_ids in zip(prompt_ids_list, target_ids_list):
            # truncate target first if over budget
            max_target = max(0, budget - len(p_ids))
            t_ids = t_ids[:max_target]
            core = p_ids + t_ids
            full = self.tokenizer.build_inputs_with_special_tokens(core)
            full_ids.append(full)
            prompt_offsets.append(num_leading)
            prompt_lens_list.append(len(p_ids))

        seq_len = max(len(x) for x in full_ids)
        text_input_ids = torch.full((B, seq_len), self.pad_token_id, dtype=torch.long)
        text_attention_mask = torch.zeros((B, seq_len), dtype=torch.long)
        for i, ids in enumerate(full_ids):
            text_input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            text_attention_mask[i, : len(ids)] = 1
        text_labels = text_input_ids.clone()
        prompt_lengths = torch.tensor(prompt_lens_list, dtype=torch.long)
        prompt_offsets_t = torch.tensor(prompt_offsets, dtype=torch.long)

        # maskable = non-pad AND not in special ids AND not in prompt prefix
        pad_mask = text_input_ids.ne(self.pad_token_id)
        maskable = pad_mask.clone()
        if self._non_maskable_ids.numel() > 0:
            special_hit = torch.isin(text_input_ids, self._non_maskable_ids)
            maskable = maskable & ~special_hit

        # Exclude leading specials + prompt tokens. Target starts at offset + prompt_length.
        positions = torch.arange(text_input_ids.size(1)).unsqueeze(0).expand_as(text_input_ids)
        prefix_end = (prompt_offsets_t + prompt_lengths).unsqueeze(1)
        prefix_mask = positions < prefix_end
        maskable = maskable & ~prefix_mask

        ratios = torch.empty(B).uniform_(self.min_ratio, self.max_ratio)
        text_mask_a, text_mask_b = sample_masks(valid_mask=maskable.to(torch.uint8), ratios=ratios)

        visual_mask_a, visual_mask_b = sample_masks(
            valid_mask=visual_attention_mask.to(torch.uint8), ratios=ratios
        )

        # ignore labels outside masked positions; per pass loss handles filtering
        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_labels": text_labels,
            "text_mask_a": text_mask_a,
            "text_mask_b": text_mask_b,
            "text_maskable": maskable.to(torch.uint8),
            "visual_features": visual_features,
            "visual_attention_mask": visual_attention_mask,
            "visual_targets": visual_targets,
            "visual_mask_a": visual_mask_a,
            "visual_mask_b": visual_mask_b,
            "corruption_ratios": ratios,
            "prompt_lengths": prompt_lengths,
        }
