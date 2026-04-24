from __future__ import annotations

import math
from typing import Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import DropPath, LayerType, Mlp, RmsNorm, use_fused_attn

from .model import SentenceTransformerTextEncoder


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DiffParallelScalingBlock(nn.Module):
    """Local variant of timm's ParallelScalingBlock with separate branch scaling."""

    fused_attn: bool

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim
        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer("qkv_bias", None)
            self.register_parameter("mlp_bias", None)
        else:
            self.register_buffer("qkv_bias", torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim, bias=proj_bias)

        if init_values is not None:
            self.ls_attn = LayerScale(dim, init_values=init_values)
            self.ls_mlp = LayerScale(dim, init_values=init_values)
        else:
            self.ls_attn = nn.Identity()
            self.ls_mlp = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, tokens, channels = x.shape
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        q = self.q_norm(q.view(batch_size, tokens, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch_size, tokens, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch_size, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(batch_size, tokens, channels)
        x_attn = self.attn_out_proj(x_attn)

        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        y = self.drop_path(self.ls_attn(x_attn) + self.ls_mlp(x_mlp))
        return x + y


class LatentPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int,
        embed_dim: int,
        tubelet_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        grid = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        x = x.flatten(2).transpose(1, 2)
        return x, grid


class LatentVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
        latent_channels: int = 16,
        latent_size: tuple[int, int] = (28, 28),
        tubelet_size: tuple[int, int, int] = (2, 4, 4),
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        init_values: float = 1e-5,
        norm_layer: LayerType = nn.LayerNorm,
        global_pool: str = "avg",
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        if global_pool not in {"avg", "token"}:
            raise ValueError(f"Unsupported global_pool={global_pool!r}")
        if tubelet_size[1] <= 0 or tubelet_size[2] <= 0:
            raise ValueError("tubelet spatial dimensions must be positive")
        if latent_size[0] % tubelet_size[1] != 0 or latent_size[1] % tubelet_size[2] != 0:
            raise ValueError("latent spatial size must be divisible by tubelet spatial size")
        self.num_frames = int(num_frames)
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.use_cls_token = use_cls_token
        self.latent_h, self.latent_w = latent_size
        self.grid_t = math.ceil(num_frames / tubelet_size[0])
        self.grid_h = self.latent_h // tubelet_size[1]
        self.grid_w = self.latent_w // tubelet_size[2]
        self.num_tokens = self.grid_t * self.grid_h * self.grid_w

        self.patch_embed = LatentPatchEmbed(
            in_chans=latent_channels,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.grid_t, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.grid_h * self.grid_w, embed_dim))
        self.pos_drop = nn.Dropout(proj_drop)

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                DiffParallelScalingBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    mlp_layer=Mlp,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    @property
    def feature_dim(self) -> int:
        return self.embed_dim

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.temporal_pos_embed, std=0.02)
        nn.init.normal_(self.spatial_pos_embed, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RmsNorm)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _pad_temporal(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        pad_t = (-x.shape[2]) % self.tubelet_size[0]
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t))
        return x, pad_t

    def _token_mask(
        self,
        num_padding_frames: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        *,
        src_t: int,
        grid_t: int,
    ) -> Optional[torch.Tensor]:
        if num_padding_frames is None:
            return None
        valid_frames = (src_t - num_padding_frames).clamp_min(0)
        valid_t = torch.div(valid_frames + self.tubelet_size[0] - 1, self.tubelet_size[0], rounding_mode="floor")
        temporal_ids = torch.arange(grid_t, device=device).unsqueeze(0)
        temporal_mask = temporal_ids < valid_t.unsqueeze(1)
        spatial_repeat = self.grid_h * self.grid_w
        token_mask = temporal_mask.unsqueeze(-1).expand(-1, -1, spatial_repeat).reshape(batch_size, -1)
        if self.cls_token is not None:
            token_mask = torch.cat([torch.ones(batch_size, 1, device=device, dtype=torch.bool), token_mask], dim=1)
        return token_mask

    @staticmethod
    def _attention_bias(token_mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if token_mask is None:
            return None
        invalid = ~token_mask
        mask_value = torch.finfo(dtype).min
        return invalid[:, None, None, :].to(dtype=dtype) * mask_value

    def _add_pos_embed(self, x: torch.Tensor, *, grid_t: int) -> torch.Tensor:
        spatial = self.spatial_pos_embed.view(1, 1, self.grid_h * self.grid_w, self.embed_dim)
        temporal = self.temporal_pos_embed[:, :grid_t].view(1, grid_t, 1, self.embed_dim)
        pos = (temporal + spatial).reshape(1, grid_t * self.grid_h * self.grid_w, self.embed_dim)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            pos = torch.cat([torch.zeros_like(cls_token), pos.expand(x.shape[0], -1, -1)], dim=1)
        else:
            pos = pos.expand(x.shape[0], -1, -1)
        return self.pos_drop(x + pos)

    def forward(
        self,
        latents: torch.Tensor,
        *,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = latents.permute(0, 2, 1, 3, 4).contiguous()
        src_t = int(x.shape[2])
        x, _pad_t = self._pad_temporal(x)
        x, grid = self.patch_embed(x)
        grid_t = grid[0]
        x = self._add_pos_embed(x, grid_t=grid_t)
        token_mask = self._token_mask(
            num_padding_frames, latents.shape[0], latents.device, src_t=src_t, grid_t=grid_t
        )
        attn_bias = self._attention_bias(token_mask, x.dtype)

        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attn_mask=attn_bias)
            if token_mask is not None:
                x = x * token_mask.unsqueeze(-1).to(x.dtype)
        x = self.norm(x)

        if self.global_pool == "token":
            if self.cls_token is None:
                raise RuntimeError("global_pool='token' requires use_cls_token=True")
            return x[:, 0]

        if token_mask is not None:
            if self.cls_token is not None:
                pooled_tokens = x[:, 1:]
                pooled_mask = token_mask[:, 1:]
            else:
                pooled_tokens = x
                pooled_mask = token_mask
            weights = pooled_mask.unsqueeze(-1).to(pooled_tokens.dtype)
            return (pooled_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        if self.cls_token is not None:
            return x[:, 1:].mean(dim=1)
        return x.mean(dim=1)


class LatentSignCLIPModel(nn.Module):
    def __init__(
        self,
        *,
        text_model_name: str = "answerdotai/ModernBERT-base",
        max_text_length: int = 16,
        embedding_dim: int = 512,
        projection_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        num_frames: int = 32,
        latent_channels: int = 16,
        latent_size: tuple[int, int] = (28, 28),
        tubelet_size: tuple[int, int, int] = (2, 4, 4),
        vision_embed_dim: int = 512,
        vision_depth: int = 8,
        vision_num_heads: int = 8,
        vision_mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 1e-5,
        loss_type: str = "infonce",
        sigmoid_bias_init: float = -10.0,
        sigmoid_logit_scale_init: float = math.log(10.0),
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        if loss_type not in {"infonce", "sigmoid"}:
            raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
        self.loss_type = loss_type
        self.video_backbone = LatentVisionTransformer(
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_size=latent_size,
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
        self.text_encoder = SentenceTransformerTextEncoder(
            text_model_name,
            max_text_length=max_text_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        text_dim = self.text_encoder.embedding_dim

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
        latents: torch.Tensor,
        *,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.video_backbone(latents, num_padding_frames=num_padding_frames)
        return F.normalize(self.video_projection(pooled), dim=-1)

    def encode_text(self, text_features: dict[str, torch.Tensor]) -> torch.Tensor:
        text_embeddings = self.text_encoder(text_features)
        return F.normalize(self.text_projection(text_embeddings), dim=-1)

    def forward(
        self,
        *,
        latents: torch.Tensor,
        text_features: dict[str, torch.Tensor],
        target_texts: Optional[list[str]] = None,
        num_padding_frames: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        video_embeddings = self.encode_video(latents, num_padding_frames=num_padding_frames)
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
            labels = -torch.ones(local_bsz, global_bsz, device=video_embeddings.device, dtype=logits_per_video.dtype)
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
                F.cross_entropy(logits_per_video, targets) +
                F.cross_entropy(logits_per_text, targets)
            )
        return {
            "loss": loss,
            "logits_per_video": logits_per_video,
            "video_embeddings": video_embeddings,
            "text_embeddings": text_embeddings,
        }


__all__ = [
    "DiffParallelScalingBlock",
    "LatentSignCLIPModel",
    "LatentVisionTransformer",
]
