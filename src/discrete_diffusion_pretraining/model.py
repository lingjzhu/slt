from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel


VisualBackbone = Callable[[torch.Tensor], torch.Tensor]
"""Callable mapping a visual input tensor to pooled per-frame features `[B, T, D]`."""


@dataclass
class DiscreteDiffusionOutput:
    loss: torch.Tensor
    text_loss: torch.Tensor
    visual_loss: torch.Tensor
    text_logits: torch.Tensor
    visual_prediction: torch.Tensor
    num_text_masked: torch.Tensor
    num_visual_masked: torch.Tensor


class DiscreteDiffusionModel(nn.Module):
    """ModernBERT multimodal denoiser over [text, pooled visual] tokens.

    The visual encoder is supplied externally (SignHiera in production) so this
    module stays decoupled from a specific backbone. The student backbone should
    map raw visual input to pooled per-frame features `[B, T, D_v]`.
    """

    def __init__(
        self,
        *,
        modernbert_name_or_path: str = "answerdotai/ModernBERT-base",
        visual_feature_dim: int,
        teacher_feature_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        mask_token_id: int,
        pad_token_id: int,
        text_loss_weight: float = 1.0,
        visual_loss_weight: float = 1.0,
        projection_dropout: float = 0.1,
        student_backbone: Optional[nn.Module] = None,
        text_backbone: Optional[nn.Module] = None,
        text_backbone_config=None,
    ) -> None:
        super().__init__()
        if text_backbone is not None:
            self.backbone_text = text_backbone
            config = text_backbone_config or getattr(text_backbone, "config", None)
            if config is None:
                raise ValueError("text_backbone must expose `.config` or `text_backbone_config` must be provided")
        else:
            config = AutoConfig.from_pretrained(modernbert_name_or_path)
            self.backbone_text = AutoModel.from_pretrained(
                modernbert_name_or_path,
                config=config,
                attn_implementation="sdpa",
            )
        self.hidden_size = config.hidden_size
        self.config = config

        self.vocab_size = vocab_size or config.vocab_size
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.text_loss_weight = float(text_loss_weight)
        self.visual_loss_weight = float(visual_loss_weight)

        self.teacher_feature_dim = int(teacher_feature_dim or visual_feature_dim)

        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.visual_mask_token = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.visual_mask_token, std=0.02)

        self.modality_embed = nn.Embedding(2, self.hidden_size)  # 0=text, 1=visual

        self.mlm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.visual_predictor_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.teacher_feature_dim),
        )

        self.student_backbone = student_backbone

    def encode_visual(self, visual_input: torch.Tensor) -> torch.Tensor:
        if self.student_backbone is None:
            # already pooled features [B, T, D_v]
            return visual_input
        return self.student_backbone(visual_input)

    def _compose_visual_tokens(
        self,
        pooled_visual: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        proj = self.visual_projector(pooled_visual)
        mask_expanded = visual_mask.unsqueeze(-1).to(proj.dtype)
        return torch.where(mask_expanded.bool(), self.visual_mask_token.to(proj.dtype), proj)

    def _embed_text(self, input_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        ids = input_ids.clone()
        ids = torch.where(text_mask.bool(), torch.full_like(ids, self.mask_token_id), ids)
        embed_layer = self.backbone_text.get_input_embeddings()
        return embed_layer(ids)

    def forward_encoder(
        self,
        *,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_corruption_mask: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        visual_corruption_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeds = self._embed_text(text_input_ids, text_corruption_mask)
        text_embeds = text_embeds + self.modality_embed.weight[0]

        visual_embeds = self._compose_visual_tokens(visual_features, visual_corruption_mask)
        visual_embeds = visual_embeds + self.modality_embed.weight[1]

        inputs_embeds = torch.cat([text_embeds, visual_embeds], dim=1)
        attention_mask = torch.cat([text_attention_mask, visual_attention_mask], dim=1)

        outputs = self.backbone_text(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state
        t_len = text_embeds.size(1)
        return hidden[:, :t_len], hidden[:, t_len:]

    def _single_pass(
        self,
        *,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_labels: torch.Tensor,
        text_corruption_mask: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        visual_targets: torch.Tensor,
        visual_corruption_mask: torch.Tensor,
    ) -> DiscreteDiffusionOutput:
        text_hidden, visual_hidden = self.forward_encoder(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_corruption_mask=text_corruption_mask,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            visual_corruption_mask=visual_corruption_mask,
        )

        text_logits = self.mlm_head(text_hidden)
        visual_prediction = self.visual_predictor_head(visual_hidden)

        # text CE only on masked positions
        text_mask_bool = text_corruption_mask.bool()
        num_text_masked = text_mask_bool.sum()
        if num_text_masked > 0:
            flat_logits = text_logits[text_mask_bool]
            flat_labels = text_labels[text_mask_bool]
            text_loss = F.cross_entropy(flat_logits, flat_labels)
        else:
            text_loss = text_logits.sum() * 0.0

        visual_mask_bool = visual_corruption_mask.bool()
        num_visual_masked = visual_mask_bool.sum()
        if num_visual_masked > 0:
            pred = visual_prediction[visual_mask_bool]
            tgt = visual_targets[visual_mask_bool]
            visual_loss = F.mse_loss(pred, tgt)
        else:
            visual_loss = visual_prediction.sum() * 0.0

        loss = self.text_loss_weight * text_loss + self.visual_loss_weight * visual_loss

        return DiscreteDiffusionOutput(
            loss=loss,
            text_loss=text_loss,
            visual_loss=visual_loss,
            text_logits=text_logits,
            visual_prediction=visual_prediction,
            num_text_masked=num_text_masked,
            num_visual_masked=num_visual_masked,
        )

    def forward(
        self,
        *,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_labels: torch.Tensor,
        text_mask_a: torch.Tensor,
        text_mask_b: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        visual_raw: Optional[torch.Tensor] = None,
        visual_attention_mask: torch.Tensor,
        visual_targets: torch.Tensor,
        visual_mask_a: torch.Tensor,
        visual_mask_b: torch.Tensor,
    ) -> dict:
        # Route visual encoding inside forward so DDP registers student_backbone
        # parameters as participating in this step (required for grad all-reduce).
        if visual_features is None:
            if visual_raw is None:
                raise ValueError("provide visual_features or visual_raw")
            visual_features = self.encode_visual(visual_raw)
        out_a = self._single_pass(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_labels=text_labels,
            text_corruption_mask=text_mask_a,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            visual_targets=visual_targets,
            visual_corruption_mask=visual_mask_a,
        )
        out_b = self._single_pass(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_labels=text_labels,
            text_corruption_mask=text_mask_b,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            visual_targets=visual_targets,
            visual_corruption_mask=visual_mask_b,
        )
        total = 0.5 * (out_a.loss + out_b.loss)
        return {
            "loss": total,
            "pass_a": out_a,
            "pass_b": out_b,
        }
