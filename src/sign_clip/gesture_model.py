from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import SentenceTransformerTextEncoder


class GestureTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        max_frames: int,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_frames = int(max_frames)
        self.embed_dim = int(embed_dim)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, self.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        gesture: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(gesture)
        x = x + self.pos_embed[:, : x.shape[1]]
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.eq(0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if attention_mask is None:
            return x.mean(dim=1)
        weights = attention_mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GestureSignCLIPModel(nn.Module):
    def __init__(
        self,
        *,
        text_model_name: str = "answerdotai/ModernBERT-base",
        max_text_length: int = 16,
        feature_dim: int = 1104,
        max_frames: int = 256,
        gesture_embed_dim: int = 512,
        gesture_depth: int = 6,
        gesture_num_heads: int = 8,
        gesture_mlp_ratio: float = 4.0,
        embedding_dim: int = 512,
        projection_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        loss_type: str = "infonce",
        sigmoid_bias_init: float = -10.0,
        sigmoid_logit_scale_init: float = math.log(10.0),
    ) -> None:
        super().__init__()
        if loss_type not in {"infonce", "sigmoid"}:
            raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
        self.loss_type = loss_type

        self.gesture_backbone = GestureTransformerBackbone(
            feature_dim=feature_dim,
            max_frames=max_frames,
            embed_dim=gesture_embed_dim,
            depth=gesture_depth,
            num_heads=gesture_num_heads,
            mlp_ratio=gesture_mlp_ratio,
            dropout=projection_dropout,
        )
        self.text_encoder = SentenceTransformerTextEncoder(
            text_model_name,
            max_text_length=max_text_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        text_dim = self.text_encoder.embedding_dim
        self.gesture_projection = nn.Sequential(
            nn.LayerNorm(gesture_embed_dim),
            nn.Linear(gesture_embed_dim, embedding_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(projection_dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        if text_dim == embedding_dim:
            self.text_projection = nn.Identity()
        else:
            self.text_projection = nn.Linear(text_dim, embedding_dim)
        if loss_type == "sigmoid":
            self.logit_scale = nn.Parameter(torch.tensor(float(sigmoid_logit_scale_init)))
            self.logit_bias = nn.Parameter(torch.tensor(float(sigmoid_bias_init)))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            self.logit_bias = None

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer

    def encode_gesture(
        self,
        gesture: torch.Tensor,
        *,
        gesture_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.gesture_backbone(gesture, attention_mask=gesture_attention_mask)
        return F.normalize(self.gesture_projection(pooled), dim=-1)

    def encode_text(self, text_features: dict[str, torch.Tensor]) -> torch.Tensor:
        text_embeddings = self.text_encoder(text_features)
        return F.normalize(self.text_projection(text_embeddings), dim=-1)

    def forward(
        self,
        *,
        gesture: torch.Tensor,
        text_features: dict[str, torch.Tensor],
        target_texts: Optional[list[str]] = None,
        gesture_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        gesture_embeddings = self.encode_gesture(gesture, gesture_attention_mask=gesture_attention_mask)
        text_embeddings = self.encode_text(text_features)

        local_bsz = gesture_embeddings.size(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        if world_size > 1:
            from torch.distributed.nn.functional import all_gather

            gathered_gesture = torch.cat(all_gather(gesture_embeddings), dim=0)
            gathered_text = torch.cat(all_gather(text_embeddings), dim=0)
            offset = torch.distributed.get_rank() * local_bsz
        else:
            gathered_gesture = gesture_embeddings
            gathered_text = text_embeddings
            offset = 0

        if self.loss_type == "sigmoid":
            logit_scale = self.logit_scale.exp()
            logits_per_gesture = logit_scale * (gesture_embeddings @ gathered_text.T) + self.logit_bias
            logits_per_text = logit_scale * (text_embeddings @ gathered_gesture.T) + self.logit_bias
        else:
            logit_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
            logits_per_gesture = logit_scale * (gesture_embeddings @ gathered_text.T)
            logits_per_text = logit_scale * (text_embeddings @ gathered_gesture.T)
        targets = torch.arange(local_bsz, device=gesture_embeddings.device) + offset

        if target_texts is not None:
            local_texts = list(target_texts)
            if world_size > 1:
                gathered = [None] * world_size
                torch.distributed.all_gather_object(gathered, local_texts)
                all_texts = [text for sub in gathered for text in sub]
            else:
                all_texts = local_texts
            local_arr = np.asarray(local_texts, dtype=object).reshape(-1, 1)
            global_arr = np.asarray(all_texts, dtype=object).reshape(1, -1)
            dup = torch.from_numpy(local_arr == global_arr).to(gesture_embeddings.device)
            rows = torch.arange(local_bsz, device=gesture_embeddings.device)
            diag = torch.zeros_like(dup)
            diag[rows, targets] = True
            false_neg = dup & ~diag
        else:
            false_neg = None

        if self.loss_type == "sigmoid":
            global_bsz = gathered_text.size(0)
            labels = -torch.ones(local_bsz, global_bsz, device=gesture_embeddings.device, dtype=logits_per_gesture.dtype)
            rows = torch.arange(local_bsz, device=gesture_embeddings.device)
            labels[rows, targets] = 1.0
            if false_neg is not None:
                mask = (~false_neg).to(logits_per_gesture.dtype)
                loss_g = -(F.logsigmoid(labels * logits_per_gesture) * mask).sum() / mask.sum().clamp_min(1.0)
                loss_t = -(F.logsigmoid(labels * logits_per_text) * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                loss_g = -F.logsigmoid(labels * logits_per_gesture).mean()
                loss_t = -F.logsigmoid(labels * logits_per_text).mean()
            loss = 0.5 * (loss_g + loss_t) * gathered_text.size(0)
        else:
            if false_neg is not None:
                neg_inf = torch.finfo(logits_per_gesture.dtype).min
                logits_per_gesture = logits_per_gesture.masked_fill(false_neg, neg_inf)
                logits_per_text = logits_per_text.masked_fill(false_neg, neg_inf)
            loss = 0.5 * (
                F.cross_entropy(logits_per_gesture, targets) +
                F.cross_entropy(logits_per_text, targets)
            )
        return {
            "loss": loss,
            "logits_per_gesture": logits_per_gesture,
            "gesture_embeddings": gesture_embeddings,
            "text_embeddings": text_embeddings,
        }


__all__ = [
    "GestureSignCLIPModel",
    "GestureTransformerBackbone",
]
