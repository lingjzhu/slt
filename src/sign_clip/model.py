from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models as st_models
from torch import nn


_PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))


_MAE_ONLY_PREFIXES = (
    "decoder_",
    "decoder.",
    "mask_token",
)


def _filter_encoder_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in _MAE_ONLY_PREFIXES)
    }


@torch.compile(fullgraph=False, dynamic=True)
def _fused_frame_pool(
    frame_features: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if frame_mask is None:
        return frame_features.mean(dim=1)
    w = frame_mask.to(frame_features.dtype).unsqueeze(-1)
    return (frame_features * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


class SignHieraMaxPooledBackbone(nn.Module):
    def __init__(self, sign_hiera: nn.Module, *, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.sign_hiera = sign_hiera
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self) -> None:
        from torch.utils.checkpoint import checkpoint

        for blk in self.sign_hiera.blocks:
            orig_forward = blk.forward

            def make_ckpt(fn):
                def wrapped(*args, **kwargs):
                    if torch.is_grad_enabled() and any(
                        torch.is_tensor(a) and a.requires_grad for a in args
                    ):
                        return checkpoint(fn, *args, use_reentrant=False, **kwargs)
                    return fn(*args, **kwargs)
                return wrapped

            blk.forward = make_ckpt(orig_forward)

    @property
    def feature_dim(self) -> int:
        return int(self.sign_hiera.blocks[-1].dim_out)

    def frame_features(
        self,
        video: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attn_mask = attention_mask
        elif padding is not None:
            attn_mask = self.sign_hiera.get_attention_mask(padding, device=video.device)
        else:
            attn_mask = None

        _, intermediates = self.sign_hiera.forward(
            video,
            attn_mask=attn_mask,
            return_intermediates=True,
        )
        return intermediates[-1].amax(dim=(2, 3))

    def forward(
        self,
        video: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.frame_features(video, attention_mask=attention_mask, padding=padding)
        return _fused_frame_pool(feats, frame_mask)


def _interpolate_temporal_pos_embed(
    state: dict[str, Any], target_len: int
) -> dict[str, Any]:
    key = "pos_embed_temporal"
    if key not in state:
        return state
    src = state[key]  # [1, T_src, C]
    if src.shape[1] == target_len:
        return state
    interp = F.interpolate(
        src.transpose(1, 2).float(), size=target_len, mode="linear", align_corners=False
    ).transpose(1, 2).to(src.dtype)
    state = dict(state)
    state[key] = interp
    return state


def build_sign_hiera_backbone(
    checkpoint_path: str | Path | None,
    *,
    model_fn: str = "hiera_base_128x224",
    map_location: str = "cpu",
    strict: bool = False,
    gradient_checkpointing: bool = False,
    num_frames: Optional[int] = None,
) -> SignHieraMaxPooledBackbone:
    from mae_pretraining.models import sign_hiera as sh
    from mae_pretraining.models.sign_hiera import SignHiera

    if num_frames is not None and model_fn == "hiera_base_128x224":
        model = SignHiera(
            num_classes=400,
            input_size=(num_frames, 224, 224),
            q_stride=(1, 2, 2),
            mask_unit_size=(1, 8, 8),
            patch_kernel=(3, 7, 7),
            patch_stride=(2, 4, 4),
            patch_padding=(1, 3, 3),
            sep_pos_embed=True,
            q_pool=3,
        )
    else:
        ctor = getattr(sh, model_fn)
        model = ctor(pretrained=False)

    if checkpoint_path is None:
        return SignHieraMaxPooledBackbone(model, gradient_checkpointing=gradient_checkpointing)

    checkpoint = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder_state = _filter_encoder_state(state)
    if num_frames is not None:
        target_t_tokens = num_frames // 2  # patch_stride temporal = 2
        encoder_state = _interpolate_temporal_pos_embed(encoder_state, target_t_tokens)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"SignHiera load mismatch: missing={missing} unexpected={unexpected}")
    return SignHieraMaxPooledBackbone(model, gradient_checkpointing=gradient_checkpointing)


class SentenceTransformerTextEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_text_length: int,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        transformer = st_models.Transformer(
            model_name_or_path,
            max_seq_length=max_text_length,
            model_args={"attn_implementation": "sdpa"},
            tokenizer_args={"use_fast": True},
        )
        pooling = st_models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        normalize = st_models.Normalize()
        self.transformer = transformer
        self.model = SentenceTransformer(modules=[transformer, pooling, normalize])
        if gradient_checkpointing and hasattr(self.transformer.auto_model, "gradient_checkpointing_enable"):
            try:
                self.transformer.auto_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.transformer.auto_model.gradient_checkpointing_enable()

    @property
    def embedding_dim(self) -> int:
        return int(self.transformer.get_word_embedding_dimension())

    @property
    def tokenizer(self):
        return self.transformer.tokenizer

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(features)["sentence_embedding"]


class SignCLIPModel(nn.Module):
    def __init__(
        self,
        *,
        hiera_checkpoint: str | Path | None,
        hiera_model_fn: str = "hiera_base_128x224",
        text_model_name: str = "answerdotai/ModernBERT-base",
        max_text_length: int = 16,
        embedding_dim: Optional[int] = None,
        projection_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        num_frames: Optional[int] = None,
        loss_type: str = "infonce",
        sigmoid_bias_init: float = -10.0,
        sigmoid_logit_scale_init: float = math.log(10.0),
    ) -> None:
        super().__init__()
        if loss_type not in ("infonce", "sigmoid"):
            raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
        self.loss_type = loss_type
        self.video_backbone = build_sign_hiera_backbone(
            hiera_checkpoint,
            model_fn=hiera_model_fn,
            gradient_checkpointing=gradient_checkpointing,
            num_frames=num_frames,
        )
        self.text_encoder = SentenceTransformerTextEncoder(
            text_model_name,
            max_text_length=max_text_length,
            gradient_checkpointing=gradient_checkpointing,
        )

        video_dim = self.video_backbone.feature_dim
        text_dim = self.text_encoder.embedding_dim
        output_dim = int(embedding_dim or text_dim)

        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, output_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(projection_dropout),
            nn.Linear(output_dim, output_dim),
        )
        if text_dim == output_dim:
            self.text_projection = nn.Identity()
        else:
            self.text_projection = nn.Linear(text_dim, output_dim)
        if loss_type == "sigmoid":
            self.logit_scale = nn.Parameter(torch.tensor(float(sigmoid_logit_scale_init)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0 if sigmoid_bias_init is None else float(sigmoid_bias_init)))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            self.logit_bias = None

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer

    def encode_video(
        self,
        video: torch.Tensor,
        *,
        video_attention_mask: Optional[torch.Tensor] = None,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.video_backbone(
            video,
            padding=num_padding_frames,
            frame_mask=video_attention_mask,
        )
        return F.normalize(self.video_projection(pooled), dim=-1)

    def encode_text(self, text_features: dict[str, torch.Tensor]) -> torch.Tensor:
        text_embeddings = self.text_encoder(text_features)
        return F.normalize(self.text_projection(text_embeddings), dim=-1)

    def forward(
        self,
        *,
        video: torch.Tensor,
        video_attention_mask: Optional[torch.Tensor],
        text_features: dict[str, torch.Tensor],
        target_texts: Optional[list[str]] = None,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        video_embeddings = self.encode_video(
            video,
            video_attention_mask=video_attention_mask,
            num_padding_frames=num_padding_frames,
        )
        text_embeddings = self.encode_text(text_features)

        local_bsz = video_embeddings.size(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        if world_size > 1:
            from torch.distributed.nn.functional import all_gather
            gathered_video = torch.cat(all_gather(video_embeddings), dim=0)
            gathered_text = torch.cat(all_gather(text_embeddings), dim=0)
            offset = torch.distributed.get_rank() * local_bsz
        else:
            gathered_video = video_embeddings
            gathered_text = text_embeddings
            offset = 0

        if self.loss_type == "sigmoid":
            logit_scale = self.logit_scale.exp()
            logits_per_video = logit_scale * (video_embeddings @ gathered_text.T) + self.logit_bias
            logits_per_text = logit_scale * (text_embeddings @ gathered_video.T) + self.logit_bias
        else:
            logit_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
            logits_per_video = logit_scale * (video_embeddings @ gathered_text.T)
            logits_per_text = logit_scale * (text_embeddings @ gathered_video.T)
        targets = torch.arange(local_bsz, device=video_embeddings.device) + offset

        if target_texts is not None:
            local_texts = list(target_texts)
            if world_size > 1:
                gathered = [None] * world_size
                torch.distributed.all_gather_object(gathered, local_texts)
                all_texts = [t for sub in gathered for t in sub]
            else:
                all_texts = local_texts
            global_bsz = len(all_texts)
            local_arr = np.asarray(local_texts, dtype=object).reshape(-1, 1)
            global_arr = np.asarray(all_texts, dtype=object).reshape(1, -1)
            dup = torch.from_numpy(local_arr == global_arr).to(video_embeddings.device)
            rows = torch.arange(local_bsz, device=video_embeddings.device)
            diag = torch.zeros_like(dup)
            diag[rows, targets] = True
            false_neg = dup & ~diag
        else:
            false_neg = None

        if self.loss_type == "sigmoid":
            global_bsz = gathered_text.size(0)
            labels_v = -torch.ones(local_bsz, global_bsz, device=video_embeddings.device, dtype=logits_per_video.dtype)
            rows = torch.arange(local_bsz, device=video_embeddings.device)
            labels_v[rows, targets] = 1.0
            labels_t = labels_v
            if false_neg is not None:
                mask = (~false_neg).to(logits_per_video.dtype)
                loss_v = -(F.logsigmoid(labels_v * logits_per_video) * mask).sum() / mask.sum().clamp_min(1.0)
                loss_t = -(F.logsigmoid(labels_t * logits_per_text) * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                loss_v = -F.logsigmoid(labels_v * logits_per_video).mean()
                loss_t = -F.logsigmoid(labels_t * logits_per_text).mean()
            loss = 0.5 * (loss_v + loss_t) * gathered_text.size(0)
        else:
            if false_neg is not None:
                neg_inf = torch.finfo(logits_per_video.dtype).min
                logits_per_video = logits_per_video.masked_fill(false_neg, neg_inf)
                logits_per_text = logits_per_text.masked_fill(false_neg, neg_inf)
            loss = 0.5 * (
                F.cross_entropy(logits_per_video, targets) +
                F.cross_entropy(logits_per_text, targets)
            )
        return {
            "loss": loss,
            "logits_per_video": logits_per_video,
            "video_embeddings": video_embeddings,
            "text_embeddings": text_embeddings,
        }
