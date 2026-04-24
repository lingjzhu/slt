"""Wrap the project's SignHiera encoder as a pooled-per-frame visual backbone.

The MAE pretraining checkpoint contains both the encoder and an MAE decoder
(`decoder_*`, `mask_token`, `decoder_pos_embed`). For stage-2 we load only the
encoder-side weights into a fresh `SignHiera` and expose a forward that returns
`[B, T_out, C]` pooled features via `extract_features`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Allow importing the project's mae_pretraining models without packaging.
_PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))


_MAE_ONLY_PREFIXES = (
    "decoder_",
    "decoder.",
    "mask_token",
)


def _filter_encoder_state(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in _MAE_ONLY_PREFIXES):
            continue
        out[k] = v
    return out


class SignHieraPooledBackbone(nn.Module):
    """Wrap SignHiera so `forward(video) -> [B, T, C]` pooled frame features."""

    def __init__(self, sign_hiera: nn.Module, use_fused: bool = False) -> None:
        super().__init__()
        self.sign_hiera = sign_hiera
        self.use_fused = use_fused

    @property
    def feature_dim(self) -> int:
        # Last stage output channels = embed_dim * 2**(num_stages-1).
        # For hiera_base_128x224 this is 768.
        return int(self.sign_hiera.blocks[-1].dim_out)

    def forward(
        self,
        video: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fn = (
            self.sign_hiera.extract_features_fused
            if self.use_fused
            else self.sign_hiera.extract_features
        )
        feats = fn(video, attention_mask=attention_mask, padding=padding)
        # extract_features flattens to [B*T, C]. Reshape back to [B, T, C].
        # With `padding`, per-sample T varies and reshape isn't valid — not supported here.
        if padding is not None:
            raise NotImplementedError(
                "SignHieraPooledBackbone.forward does not support padding= "
                "(variable per-sample T breaks [B,T,C] reshape)."
            )
        B = video.shape[0]
        if feats.shape[0] % B != 0:
            raise RuntimeError(f"unexpected feats shape {feats.shape} for batch {B}")
        return feats.view(B, feats.shape[0] // B, feats.shape[-1])


def build_sign_hiera_student(
    checkpoint_path: str | Path,
    *,
    model_fn: str = "hiera_base_128x224",
    map_location: str = "cpu",
    use_fused: bool = False,
    strict: bool = False,
) -> SignHieraPooledBackbone:
    """Build a SignHiera student wrapper and load encoder weights from an MAE checkpoint."""
    from mae_pretraining.models import sign_hiera as sh

    ctor = getattr(sh, model_fn)
    model = ctor(pretrained=False)

    ckpt = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    enc_state = _filter_encoder_state(state)
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"SignHiera load mismatch: missing={missing} unexpected={unexpected}")
    return SignHieraPooledBackbone(model, use_fused=use_fused)
