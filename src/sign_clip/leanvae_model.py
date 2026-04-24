from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .latent_model import DiffParallelScalingBlock, LatentVisionTransformer
from .model import SentenceTransformerTextEncoder

_LEANVAE_ROOT = Path("/home/slimelab/Projects/Sign/LeanVAE")
if str(_LEANVAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEANVAE_ROOT))


class LeanVAEVisualEncoder(nn.Module):
    """Wraps LeanVAE's DWT + encoder to extract pre-bottleneck features.

    Returns the encoder output ``p`` *before* the ISTA latent bottleneck,
    giving a spatiotemporal token grid of shape ``[B, T_enc, H_enc, W_enc, D]``
    where ``D = l_dim + h_dim`` (default 512).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from LeanVAE import LeanVAE

        vae = LeanVAE.load_from_checkpoint(str(checkpoint_path), strict=False)
        self.dwt = vae.dwt
        self.encoder = vae.encoder
        self._encoder_dim = int(vae.args.l_dim + vae.args.h_dim)

        # drop decoder / bottleneck — not needed
        del vae

        if freeze:
            self.freeze()

    @property
    def feature_dim(self) -> int:
        return self._encoder_dim

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: ``[B, 3, T, H, W]`` in ``[-0.5, 0.5]`` range.
                   ``T`` must satisfy ``(T - 1) % 4 == 0``.

        Returns:
            Encoder features ``p`` of shape ``[B, T_enc, H_enc, W_enc, D]``
            taken *before* ``latent_bottleneck.sample`` (z_mean projection) and
            ``std_layer`` (z_std projection). Deterministic — no reparam sampling.
        """
        x_dwt = self.dwt(video)
        p = self.encoder.encode(x=x_dwt)
        return p


def _compute_encoder_padding(
    num_padding_frames: torch.Tensor,
    total_raw_frames: int,
) -> torch.Tensor:
    """Map raw-video padding counts to encoder temporal-token padding counts.

    The LeanVAE DWT + encoder pipeline reduces temporal resolution:
      raw T  ->  DWT video part (T-1)/2  ->  patch stride 2  ->  (T-1)/4 tokens
    Plus 1 image token (always valid).  Total encoder tokens = 1 + (T-1)/4.

    Padding frames in the raw video translate to padding tokens in the encoder
    output via: ``enc_padding = floor(raw_padding / 4)``.
    """
    return torch.div(num_padding_frames, 4, rounding_mode="floor")


class LeanVAECLIPModel(nn.Module):
    """CLIP model using frozen LeanVAE encoder features + latent ViT head.

    Visual path:
        raw video -> LeanVAE DWT + encoder -> latent ViT -> pooling -> projection
    Text path:
        tokens -> ModernBERT sentence encoder -> projection
    """

    def __init__(
        self,
        *,
        leanvae_checkpoint: str | Path,
        freeze_encoder: bool = True,
        text_model_name: str = "answerdotai/ModernBERT-base",
        max_text_length: int = 16,
        embedding_dim: int = 512,
        projection_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        # ViT head config
        num_frames: int = 65,
        encoder_spatial_size: tuple[int, int] = (28, 28),
        tubelet_size: tuple[int, int, int] = (1, 4, 4),
        vision_embed_dim: int = 512,
        vision_depth: int = 6,
        vision_num_heads: int = 8,
        vision_mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 1e-5,
        global_pool: str = "avg",
        # loss
        loss_type: str = "infonce",
        sigmoid_bias_init: float = -10.0,
        sigmoid_logit_scale_init: float = math.log(10.0),
    ) -> None:
        super().__init__()
        if loss_type not in {"infonce", "sigmoid"}:
            raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
        self.loss_type = loss_type
        self.num_raw_frames = num_frames

        # --- LeanVAE encoder (frozen by default) ---
        self.leanvae_encoder = LeanVAEVisualEncoder(
            leanvae_checkpoint, freeze=freeze_encoder,
        )
        encoder_dim = self.leanvae_encoder.feature_dim  # 512

        # --- optional linear adapter if encoder dim != ViT embed dim ---
        if encoder_dim != vision_embed_dim:
            self.adapter = nn.Linear(encoder_dim, vision_embed_dim)
        else:
            self.adapter = nn.Identity()

        # Encoder temporal tokens: 1 (image) + (num_frames-1)/4 (video)
        enc_temporal_tokens = 1 + (num_frames - 1) // 4

        # --- latent ViT head ---
        # LatentVisionTransformer expects input [B, T, C, H, W]
        # We'll feed encoder features rearranged to that format
        self.video_backbone = LatentVisionTransformer(
            num_frames=enc_temporal_tokens,
            latent_channels=vision_embed_dim,
            latent_size=encoder_spatial_size,
            tubelet_size=tubelet_size,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_ratio=vision_mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            init_values=init_values,
            global_pool=global_pool,
        )

        # --- text encoder ---
        self.text_encoder = SentenceTransformerTextEncoder(
            text_model_name,
            max_text_length=max_text_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        text_dim = self.text_encoder.embedding_dim

        # --- projection heads ---
        self.video_projection = nn.Sequential(
            nn.LayerNorm(self.video_backbone.feature_dim),
            nn.Linear(self.video_backbone.feature_dim, embedding_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(projection_dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        if text_dim == embedding_dim:
            self.text_projection = nn.Identity()
        else:
            self.text_projection = nn.Linear(text_dim, embedding_dim)

        # --- logit scale ---
        if loss_type == "sigmoid":
            self.logit_scale = nn.Parameter(torch.tensor(float(sigmoid_logit_scale_init)))
            self.logit_bias = nn.Parameter(torch.tensor(float(sigmoid_bias_init)))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            self.logit_bias = None

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer

    def encode_video(
        self,
        video: torch.Tensor,
        *,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode raw video to normalised CLIP embedding.

        Args:
            video: ``[B, 3, T, H, W]``. Accepts either float in ``[-0.5, 0.5]``
                or uint8 in ``[0, 255]`` — in the latter case the dataloader
                emitted raw pixel values and we do the float cast + LeanVAE
                normalization (``x/255 - 0.5``) here, on-device.
            num_padding_frames: per-sample raw-video padding count.
        """
        if video.dtype == torch.uint8:
            # Fused on-GPU normalize: uint8 -> bf16/float, /255, -0.5.
            # Much faster than CPU tensor_normalize + halves H2D traffic.
            video = video.to(dtype=torch.float32).mul_(1.0 / 255.0).sub_(0.5)
        # encoder features: [B, T_enc, H_enc, W_enc, D]
        enc_features = self.leanvae_encoder(video)
        # optional adapter
        enc_features = self.adapter(enc_features)
        # rearrange to [B, T_enc, D, H_enc, W_enc] for LatentVisionTransformer
        enc_features = rearrange(enc_features, "b t h w d -> b t d h w")

        enc_padding = None
        if num_padding_frames is not None:
            enc_padding = _compute_encoder_padding(
                num_padding_frames, self.num_raw_frames,
            )

        pooled = self.video_backbone(enc_features, num_padding_frames=enc_padding)
        return F.normalize(self.video_projection(pooled), dim=-1)

    def encode_text(self, text_features: dict[str, torch.Tensor]) -> torch.Tensor:
        text_embeddings = self.text_encoder(text_features)
        return F.normalize(self.text_projection(text_embeddings), dim=-1)

    def forward(
        self,
        *,
        video: torch.Tensor,
        text_features: dict[str, torch.Tensor],
        target_texts: Optional[list[str]] = None,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        video_embeddings = self.encode_video(video, num_padding_frames=num_padding_frames)
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
                all_texts = [text for sub in gathered for text in sub]
            else:
                all_texts = local_texts
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
            labels = -torch.ones(
                local_bsz, global_bsz,
                device=video_embeddings.device, dtype=logits_per_video.dtype,
            )
            rows = torch.arange(local_bsz, device=video_embeddings.device)
            labels[rows, targets] = 1.0
            if false_neg is not None:
                mask = (~false_neg).to(logits_per_video.dtype)
                loss_v = -(F.logsigmoid(labels * logits_per_video) * mask).sum() / mask.sum().clamp_min(1.0)
                loss_t = -(F.logsigmoid(labels * logits_per_text) * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                loss_v = -F.logsigmoid(labels * logits_per_video).mean()
                loss_t = -F.logsigmoid(labels * logits_per_text).mean()
            loss = 0.5 * (loss_v + loss_t) * gathered_text.size(0)
        else:
            if false_neg is not None:
                neg_inf = torch.finfo(logits_per_video.dtype).min
                logits_per_video = logits_per_video.masked_fill(false_neg, neg_inf)
                logits_per_text = logits_per_text.masked_fill(false_neg, neg_inf)
            loss = 0.5 * (
                F.cross_entropy(logits_per_video, targets)
                + F.cross_entropy(logits_per_text, targets)
            )
        return {
            "loss": loss,
            "logits_per_video": logits_per_video,
            "video_embeddings": video_embeddings,
            "text_embeddings": text_embeddings,
        }


__all__ = [
    "LeanVAECLIPModel",
    "LeanVAEVisualEncoder",
]
