from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gesture_encoder import GestureEncoder


# ── Masking ────────────────────────────────────────────────────────────────────

def sample_token_span_mask(
    B: int,
    T2: int,
    mask_ratio: float = 0.5,
    min_span_frames: int = 4,
    device: torch.device = torch.device("cpu"),
    valid_frame_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns (B, T2) bool mask where True = downsampled token is masked.

    `min_span_frames` stays in original frame units for the CLI/config. Since
    each downsampled token covers two source frames, it is converted to token
    units before sampling.
    """
    masks = []
    for b in range(B):
        valid_frames = T2 * 2 if valid_frame_lengths is None else int(valid_frame_lengths[b].item())
        valid_tokens = min(T2, valid_frames // 2)
        mask = torch.zeros(T2, dtype=torch.bool, device=device)
        if valid_tokens <= 0:
            masks.append(mask)
            continue

        min_span_tokens = max(1, math.ceil(min_span_frames / 2))
        target_masked = min(valid_tokens, max(1, int(valid_tokens * mask_ratio)))
        max_span = min(min_span_tokens * 2 - 1, valid_tokens)
        min_cur_span = min(min_span_tokens, valid_tokens)
        n_masked = 0
        attempts = 0
        while n_masked < target_masked and attempts < valid_tokens * 4:
            attempts += 1
            span = min_cur_span + torch.randint(
                0,
                max(1, max_span - min_cur_span + 1),
                (1,),
                device=device,
            ).item()
            start = torch.randint(0, max(1, valid_tokens - span + 1), (1,), device=device).item()
            mask[start: start + span] = True
            n_masked = int(mask.sum())
        masks.append(mask)
    return torch.stack(masks, dim=0)


# ── SwiGLU projection ─────────────────────────────────────────────────────────

class SwiGLUProjection(nn.Module):
    """Lightweight SwiGLU MLP to project gesture pooled embedding to text embedding space."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.wi = nn.Linear(in_dim, hidden_dim * 2, bias=False)
        self.wo = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate, hidden = self.wi(x).chunk(2, dim=-1)
        return self.wo(F.silu(gate) * hidden)


# ── InfoNCE loss with all-gather ───────────────────────────────────────────────

def infonce_loss(
    gesture_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: Optional[torch.Tensor] = None,
    loss_type: str = "infonce",
    target_texts: Optional[list[str]] = None,
) -> torch.Tensor:
    """
    Symmetric contrastive loss.
    Gathers embeddings across all DDP ranks before computing the loss so that
    every device sees the full global batch. Duplicate captions are masked as
    false negatives, matching the sign_clip gesture training path.
    """
    if loss_type not in {"infonce", "sigmoid"}:
        raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
    if gesture_emb.dtype != text_emb.dtype:
        common = torch.promote_types(gesture_emb.dtype, text_emb.dtype)
        gesture_emb = gesture_emb.to(common)
        text_emb = text_emb.to(common)

    local_bsz = gesture_emb.size(0)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_gather

        all_gesture = torch.cat(all_gather(gesture_emb), dim=0)
        all_text = torch.cat(all_gather(text_emb), dim=0)
        offset = torch.distributed.get_rank() * local_bsz
    else:
        all_gesture = gesture_emb
        all_text = text_emb
        offset = 0

    targets = torch.arange(local_bsz, device=gesture_emb.device) + offset

    false_neg = None
    if target_texts is not None:
        local_texts = list(target_texts)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local_texts)
            all_texts = [text for sub in gathered for text in sub]
        else:
            all_texts = local_texts
        local_arr = np.asarray(local_texts, dtype=object).reshape(-1, 1)
        global_arr = np.asarray(all_texts, dtype=object).reshape(1, -1)
        dup = torch.from_numpy(local_arr == global_arr).to(gesture_emb.device)
        rows = torch.arange(local_bsz, device=gesture_emb.device)
        diag = torch.zeros_like(dup)
        diag[rows, targets] = True
        false_neg = dup & ~diag

    if loss_type == "sigmoid":
        if logit_bias is None:
            raise ValueError("sigmoid contrastive loss requires logit_bias")
        logits_g2t = logit_scale.exp() * (gesture_emb @ all_text.T) + logit_bias
        logits_t2g = logit_scale.exp() * (text_emb @ all_gesture.T) + logit_bias
        labels = -torch.ones(
            local_bsz,
            all_text.size(0),
            device=gesture_emb.device,
            dtype=logits_g2t.dtype,
        )
        rows = torch.arange(local_bsz, device=gesture_emb.device)
        labels[rows, targets] = 1.0
        if false_neg is not None:
            mask = (~false_neg).to(logits_g2t.dtype)
            loss_g = -(F.logsigmoid(labels * logits_g2t) * mask).sum() / mask.sum().clamp_min(1.0)
            loss_t = -(F.logsigmoid(labels * logits_t2g) * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss_g = -F.logsigmoid(labels * logits_g2t).mean()
            loss_t = -F.logsigmoid(labels * logits_t2g).mean()
        loss = 0.5 * (loss_g + loss_t) * all_text.size(0)
    else:
        logits_g2t = logit_scale.clamp(max=math.log(100.0)).exp() * (gesture_emb @ all_text.T)
        logits_t2g = logit_scale.clamp(max=math.log(100.0)).exp() * (text_emb @ all_gesture.T)
        if false_neg is not None:
            neg_inf = torch.finfo(logits_g2t.dtype).min
            logits_g2t = logits_g2t.masked_fill(false_neg, neg_inf)
            logits_t2g = logits_t2g.masked_fill(false_neg, neg_inf)
        loss = 0.5 * (F.cross_entropy(logits_g2t, targets) + F.cross_entropy(logits_t2g, targets))
    return loss


# ── Full pretraining model ────────────────────────────────────────────────────

class GesturePretrainModel(nn.Module):
    """
    Gesture-language contrastive pretraining model with masked reconstruction.

    The text encoder is NOT part of this module — it is held externally by the
    trainer (frozen, not wrapped in DDP) and text embeddings are passed in
    pre-computed. This lets the DDP wrapper run with
    `find_unused_parameters=False` for lower overhead.

    Two losses, computed from ONE forward pass through the gesture encoder:
    1. InfoNCE contrastive loss between the masked-input pooled gesture embedding
       and the pre-computed text embedding. Masking acts as augmentation — this
       is the standard recipe in SimMIM / masked-CLIP.
    2. MSE masked reconstruction: a lightweight MLP decoder predicts the two
       original frames covered by each downsampled token.
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
        gesture_attn_implementation: str = "flash_attention_2",
        gesture_hidden_activation: str = "silu",
        # Projection
        text_embed_dim: int = 768,
        proj_hidden_dim: Optional[int] = None,
        # Losses
        temperature: float = 0.05,
        loss_type: str = "infonce",
        sigmoid_bias_init: float = -10.0,
        sigmoid_logit_scale_init: float = math.log(10.0),
        mask_ratio: float = 0.5,
        min_span: int = 4,
        recon_weight: float = 1.0,
        no_mae: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.config = {
            "model_type": "gesture_encoder_with_projection",
            "architecture": "GestureEncoderWithProjection",
            "in_dim": int(in_dim),
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_hidden_layers),
            "num_attention_heads": int(num_attention_heads),
            "intermediate_size": int(intermediate_size),
            "max_position_embeddings": int(max_position_embeddings),
            "global_attn_every_n_layers": int(global_attn_every_n_layers),
            "local_attention": int(local_attention),
            "gesture_attn_implementation": gesture_attn_implementation,
            "gesture_hidden_activation": gesture_hidden_activation,
            "text_embed_dim": int(text_embed_dim),
            "proj_hidden_dim": int(proj_hidden_dim or hidden_size),
            "loss_type": loss_type,
            "temperature": float(temperature),
        }
        if loss_type not in {"infonce", "sigmoid"}:
            raise ValueError(f"loss_type must be 'infonce' or 'sigmoid', got {loss_type}")
        self.loss_type = loss_type
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.recon_weight = recon_weight
        self.no_mae = no_mae

        self.gesture_encoder = GestureEncoder(
            in_dim=in_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            global_attn_every_n_layers=global_attn_every_n_layers,
            local_attention=local_attention,
            attn_implementation=gesture_attn_implementation,
            hidden_activation=gesture_hidden_activation,
        )

        self.gesture_proj = SwiGLUProjection(
            in_dim=hidden_size,
            out_dim=text_embed_dim,
            hidden_dim=proj_hidden_dim or hidden_size,
        )

        # Reconstruction decoder: predicts two original frames (pair) per downsampled token
        self.recon_decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, in_dim * 2, bias=True),
        )
        if no_mae:
            for p in self.recon_decoder.parameters():
                p.requires_grad_(False)
        if loss_type == "sigmoid":
            self.logit_scale = nn.Parameter(torch.tensor(float(sigmoid_logit_scale_init)))
            self.logit_bias = nn.Parameter(torch.tensor(float(sigmoid_bias_init)))
        else:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(temperature))))
            self.logit_bias = None

    def encode_gesture(
        self,
        gesture: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns L2-normalised projected gesture embedding (B, text_dim)."""
        enc = self.gesture_encoder(gesture, attention_mask=attention_mask)
        emb = self.gesture_proj(enc["pooled"])
        return F.normalize(emb, dim=-1)

    def gesture_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("gesture_encoder.", self.gesture_encoder),
            ("gesture_proj.", self.gesture_proj),
        ):
            for key, value in module.state_dict().items():
                state[f"{prefix}{key}"] = value.detach().cpu().clone()
        state["logit_scale"] = self.logit_scale.detach().cpu().clone()
        if self.logit_bias is not None:
            state["logit_bias"] = self.logit_bias.detach().cpu().clone()
        return state

    def save_pretrained(self, output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        weights_path = output_dir / "pytorch_model.bin"
        config_path.write_text(json.dumps(self.config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        torch.save(self.gesture_encoder_state_dict(), weights_path)
        return {"config": config_path, "weights": weights_path}

    def forward(
        self,
        gesture: torch.Tensor,
        text_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_texts: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        One forward pass through the gesture encoder with masking applied.
        Both the contrastive loss and the reconstruction loss are computed
        from the same encoder output — ~2× fewer FLOPs than the previous
        two-pass version.

        Args:
            gesture:        (B, T, in_dim)
            text_emb:       (B, text_embed_dim) pre-computed L2-normalised
                            text embeddings from the external frozen encoder.
            attention_mask: (B, T) long, 1=valid frame.

        Returns dict with 'loss', 'loss_contrastive', 'loss_recon'.
        """
        B, T, _ = gesture.shape
        device = gesture.device

        # Pad sequence length to an even number so conv stride=2 is exact
        if T % 2 == 1:
            gesture = F.pad(gesture, (0, 0, 0, 1), value=0.0)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, 1), value=0)
            T = T + 1

        frame_mask = None
        if not self.no_mae:
            valid_lengths = (
                attention_mask.bool().sum(dim=1)
                if attention_mask is not None
                else torch.full((B,), T, dtype=torch.long, device=device)
            )

            # Mask downsampled tokens, then expand selected tokens back to the two
            # source frames they reconstruct.
            token_mask = sample_token_span_mask(
                B,
                T // 2,
                self.mask_ratio,
                self.min_span,
                device,
                valid_frame_lengths=valid_lengths,
            )
            frame_mask = token_mask.repeat_interleave(2, dim=1)

        # ── Single masked encoder forward ──────────────────────────────────
        enc = self.gesture_encoder(
            gesture,
            attention_mask=attention_mask,
            frame_mask=frame_mask,
        )
        token_feats = enc["token_features"]  # (B, T//2, hidden_size)
        T2 = token_feats.shape[1]

        # ── Contrastive loss (gesture ↔ text) ──────────────────────────────
        # Pooled embedding comes from the masked input — masking doubles as augmentation.
        gesture_emb = F.normalize(self.gesture_proj(enc["pooled"]), dim=-1)
        loss_contrast = infonce_loss(
            gesture_emb,
            text_emb,
            logit_scale=self.logit_scale,
            logit_bias=self.logit_bias,
            loss_type=self.loss_type,
            target_texts=target_texts,
        )

        if self.no_mae:
            loss_recon = gesture.new_zeros(())
        else:
            # ── Reconstruction loss ────────────────────────────────────────
            recon = self.recon_decoder(token_feats)                # (B, T//2, in_dim * 2)
            recon = recon.view(B, T2, 2, self.in_dim)              # (B, T//2, 2, in_dim)
            target = gesture.view(B, T2, 2, self.in_dim)           # (B, T//2, 2, in_dim)

            assert frame_mask is not None
            fm = frame_mask.view(B, T2, 2)                         # (B, T//2, 2)

            if fm.any():
                loss_recon = F.mse_loss(recon[fm], target[fm])
            else:
                loss_recon = gesture.new_zeros(())

        loss = loss_contrast + self.recon_weight * loss_recon

        return {
            "loss": loss,
            "loss_contrastive": loss_contrast.detach(),
            "loss_recon": loss_recon.detach(),
        }


class GestureEncoderWithProjection(nn.Module):
    """Standalone pretrained gesture encoder: backbone plus projection head."""

    def __init__(
        self,
        *,
        in_dim: int = 1104,
        hidden_size: int = 768,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        intermediate_size: int = 1152,
        max_position_embeddings: int = 8192,
        global_attn_every_n_layers: int = 3,
        local_attention: int = 128,
        gesture_attn_implementation: str = "flash_attention_2",
        gesture_hidden_activation: str = "silu",
        text_embed_dim: int = 768,
        proj_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = {
            "model_type": "gesture_encoder_with_projection",
            "architecture": self.__class__.__name__,
            "in_dim": int(in_dim),
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_hidden_layers),
            "num_attention_heads": int(num_attention_heads),
            "intermediate_size": int(intermediate_size),
            "max_position_embeddings": int(max_position_embeddings),
            "global_attn_every_n_layers": int(global_attn_every_n_layers),
            "local_attention": int(local_attention),
            "gesture_attn_implementation": gesture_attn_implementation,
            "gesture_hidden_activation": gesture_hidden_activation,
            "text_embed_dim": int(text_embed_dim),
            "proj_hidden_dim": int(proj_hidden_dim or hidden_size),
        }
        self.gesture_encoder = GestureEncoder(
            in_dim=in_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            global_attn_every_n_layers=global_attn_every_n_layers,
            local_attention=local_attention,
            attn_implementation=gesture_attn_implementation,
            hidden_activation=gesture_hidden_activation,
        )
        self.gesture_proj = SwiGLUProjection(
            in_dim=hidden_size,
            out_dim=text_embed_dim,
            hidden_dim=proj_hidden_dim or hidden_size,
        )

    def forward(
        self,
        gesture: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc = self.gesture_encoder(gesture, attention_mask=attention_mask)
        return F.normalize(self.gesture_proj(enc["pooled"]), dim=-1)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "GestureEncoderWithProjection":
        path = Path(pretrained_model_name_or_path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        kwargs = {
            key: config[key]
            for key in (
                "in_dim",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "max_position_embeddings",
                "global_attn_every_n_layers",
                "local_attention",
                "gesture_attn_implementation",
                "gesture_hidden_activation",
                "text_embed_dim",
                "proj_hidden_dim",
            )
            if key in config
        }
        model = cls(**kwargs)
        state = torch.load(str(path / "pytorch_model.bin"), map_location=map_location, weights_only=False)
        state = {
            key: value
            for key, value in state.items()
            if key.startswith("gesture_encoder.") or key.startswith("gesture_proj.")
        }
        model.load_state_dict(state, strict=strict)
        return model
