from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
from transformers import AutoConfig, ModernBertModel


class ConvBackbone(nn.Module):
    """
    Replaces the embedding lookup in ModernBERT.
    Projects raw gesture features to hidden_size and downsamples 2x temporally.
    """

    def __init__(self, in_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        # (B, in_dim, T) → (B, hidden_size, T//2)
        self.conv = nn.Conv1d(in_dim, hidden_size, kernel_size=3, stride=2, padding=1, bias=False)
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        x = self.in_norm(x)
        x = x.transpose(1, 2)          # → (B, in_dim, T)
        x = self.conv(x)               # → (B, hidden_size, T//2)
        x = x.transpose(1, 2)          # → (B, T//2, hidden_size)
        x = self.out_norm(x)
        return x


class GestureEncoder(nn.Module):
    """
    ModernBERT base transformer encoder with a 1D conv backbone (from scratch, no pretrained weights).

    Input:  gesture features (B, T, in_dim)
    Output: token sequence (B, T//2, hidden_size) and pooled (B, hidden_size)
    """

    def __init__(
        self,
        in_dim: int = 1104,
        hidden_size: int = 768,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        intermediate_size: int = 1152,
        max_position_embeddings: int = 8192,
        global_attn_every_n_layers: int = 3,
        local_attention: int = 128,
        attn_implementation: str = "flash_attention_2",
        hidden_activation: str = "silu",
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size

        # Learnable mask token used for masked reconstruction
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

        self.backbone = ConvBackbone(in_dim, hidden_size)

        layer_types = [
            "full_attention" if i % max(1, global_attn_every_n_layers) == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        bert_cfg = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        bert_cfg.hidden_size = hidden_size
        bert_cfg.num_hidden_layers = num_hidden_layers
        bert_cfg.num_attention_heads = num_attention_heads
        bert_cfg.intermediate_size = intermediate_size
        bert_cfg.hidden_activation = hidden_activation
        # Liger MLP modules follow the common HF `hidden_act` naming.
        bert_cfg.hidden_act = hidden_activation
        bert_cfg.max_position_embeddings = max_position_embeddings
        bert_cfg.layer_types = layer_types
        bert_cfg.local_attention = local_attention
        bert_cfg.rope_parameters = {}
        if "sliding_attention" in layer_types:
            bert_cfg.rope_parameters["sliding_attention"] = {
                "rope_type": "default",
                "rope_theta": 10000.0,
            }
        if "full_attention" in layer_types:
            bert_cfg.rope_parameters["full_attention"] = {
                "rope_type": "default",
                "rope_theta": 160000.0,
            }
        bert_cfg.attn_implementation = attn_implementation
        bert_cfg._attn_implementation = attn_implementation
        bert_cfg.dtype = torch.bfloat16
        bert_cfg.reference_compile = False
        self.transformer = ModernBertModel(bert_cfg)

    @staticmethod
    def _pad_to_even(x: torch.Tensor, dim: int = 1, value: float = 0.0) -> torch.Tensor:
        T = x.shape[dim]
        if T % 2 == 0:
            return x
        pad_shape = [0] * (2 * x.ndim)
        # F.pad expects reversed dim order; set the "after" pad for `dim`
        pad_shape[2 * (x.ndim - 1 - dim) + 1] = 1
        return torch.nn.functional.pad(x, pad_shape, value=value)

    def _downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B, T_even) long (T_even is even). Downsample to (B, T_even//2).
        T2 = mask.shape[1] // 2
        m = mask.view(mask.shape[0], T2, 2)
        return m.min(dim=-1).values

    def forward(
        self,
        gesture: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            gesture:        (B, T, in_dim) raw gesture features
            attention_mask: (B, T) long, 1=valid frame (padding mask)
            frame_mask:     (B, T) bool, True=mask this frame for reconstruction

        Returns dict with:
            token_features: (B, T//2, hidden_size)
            pooled:         (B, hidden_size) mean-pooled over valid tokens
            ds_attn_mask:   (B, T//2) downsampled attention mask
        """
        B, T, _ = gesture.shape

        # Pad input sequence to an even length so conv stride=2 is exact
        gesture = self._pad_to_even(gesture, dim=1, value=0.0)
        if attention_mask is not None:
            attention_mask = self._pad_to_even(attention_mask, dim=1, value=0)
        if frame_mask is not None:
            frame_mask = self._pad_to_even(frame_mask, dim=1, value=False)

        # Apply mask token replacement before conv
        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1).to(gesture.dtype)
            masked_gesture = gesture * (1 - mask_expanded) + self.mask_token * mask_expanded
        else:
            masked_gesture = gesture

        token_features = self.backbone(masked_gesture)  # (B, T//2, hidden_size)

        # Downsample attention mask
        if attention_mask is not None:
            ds_attn_mask = self._downsample_mask(attention_mask)
        else:
            T2 = token_features.shape[1]
            ds_attn_mask = torch.ones(B, T2, dtype=torch.long, device=gesture.device)

        out = self.transformer(
            inputs_embeds=token_features,
            attention_mask=ds_attn_mask,
        )
        hidden = out.last_hidden_state  # (B, T//2, hidden_size)

        # Masked mean pooling
        weights = ds_attn_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return {
            "token_features": hidden,
            "pooled": pooled,
            "ds_attn_mask": ds_attn_mask,
        }
