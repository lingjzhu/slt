from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from gesture_pretraining.models.gesture_encoder import GestureEncoder

logger = logging.getLogger(__name__)


def _swap_layernorms_to_liger(module: nn.Module, liger_cls: type) -> int:
    """Recursively replace ``nn.LayerNorm`` modules with a Liger fused version."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm) and not isinstance(child, liger_cls):
            new = liger_cls(
                hidden_size=child.normalized_shape[0],
                eps=child.eps,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None and getattr(new, "bias", None) is not None:
                    new.bias.copy_(child.bias)
            new.to(child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new)
            n += 1
        else:
            n += _swap_layernorms_to_liger(child, liger_cls)
    return n


def _t5_fused_loss(t5: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the LM cross-entropy loss with the Triton fused CE kernel."""
    from .kernels import fast_cross_entropy_loss
    return fast_cross_entropy_loss(logits, labels)


class GestureSignT5(nn.Module):
    """T5 conditioned on hidden states from a pretrained GestureEncoder.

    Pipeline:
        gesture features (B, T, in_dim)
          -> GestureEncoder -> (B, T//2, hidden_size)
          -> MLP projector -> (B, T//2, d_model)
          -> [prompt embeds; projected gesture] -> T5 encoder -> T5 decoder
    """

    main_input_name = "input_features"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        gesture_checkpoint: str | Path | None = None,
        gesture_in_dim: int = 1104,
        gesture_hidden_size: int = 768,
        gesture_num_hidden_layers: int = 22,
        gesture_num_attention_heads: int = 12,
        gesture_intermediate_size: int = 1152,
        gesture_max_position_embeddings: int = 1024,
        gesture_global_attn_every_n_layers: int = 3,
        gesture_local_attention: int = 128,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.1,
        freeze_gesture_encoder: bool = False,
        freeze_t5: bool = False,
        attn_implementation: str = "sdpa",
        use_efficient_kernels: bool = True,
    ) -> None:
        super().__init__()

        load_kwargs: dict[str, Any] = {}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
        except (TypeError, ValueError) as exc:
            if "attn_implementation" in str(exc) or "scaled_dot_product_attention" in str(exc):
                self.t5 = T5ForConditionalGeneration.from_pretrained(
                    model_name_or_path, attn_implementation="eager"
                )
            else:
                raise
        self.config = self.t5.config
        d_model = self.config.d_model

        if use_efficient_kernels:
            try:
                from .patching import patch_t5_for_efficiency
                stats = patch_t5_for_efficiency(
                    self.t5,
                    sdpa_attention=True,
                    flex_attention=False,
                    compile=False,
                )
                logger.info("Patched T5 with efficient kernels: %s", stats)
            except Exception as exc:
                logger.warning("Skipped T5 efficiency patches (%s)", exc)
            try:
                from .kernels import fast_cross_entropy_loss as _ce  # noqa: F401
                self._fused_ce_enabled = True
                logger.info("Triton fused cross entropy enabled for T5 loss")
            except Exception as exc:
                self._fused_ce_enabled = False
                logger.warning("Triton fused CE unavailable (%s); falling back to nn.CrossEntropyLoss", exc)
        else:
            self._fused_ce_enabled = False

        self.gesture_encoder = GestureEncoder(
            in_dim=gesture_in_dim,
            hidden_size=gesture_hidden_size,
            num_hidden_layers=gesture_num_hidden_layers,
            num_attention_heads=gesture_num_attention_heads,
            intermediate_size=gesture_intermediate_size,
            max_position_embeddings=gesture_max_position_embeddings,
            global_attn_every_n_layers=gesture_global_attn_every_n_layers,
            local_attention=gesture_local_attention,
        )

        if gesture_checkpoint:
            self._load_gesture_checkpoint(gesture_checkpoint)

        self._freeze_unused_translation_parameters()

        if use_efficient_kernels:
            try:
                from liger_kernel.transformers import LigerLayerNorm
                n = _swap_layernorms_to_liger(self.gesture_encoder, LigerLayerNorm)
                logger.info("Swapped %d gesture-encoder LayerNorms to LigerLayerNorm", n)
            except Exception as exc:
                logger.warning("Liger LayerNorm swap skipped (%s)", exc)

        if freeze_gesture_encoder:
            for p in self.gesture_encoder.parameters():
                p.requires_grad_(False)
            self.gesture_encoder.eval()
        self._frozen_gesture = freeze_gesture_encoder

        if freeze_t5:
            for p in self.t5.parameters():
                p.requires_grad_(False)
            self.t5.eval()
        self._frozen_t5 = freeze_t5

        proj_hidden = projection_hidden_dim or d_model
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(gesture_hidden_size),
            nn.Linear(gesture_hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(proj_hidden, d_model),
            nn.Dropout(projection_dropout),
        )

    def _freeze_unused_translation_parameters(self) -> None:
        """Freeze gesture-pretraining-only parameters unused by SLT forward.

        Translation always feeds continuous ``inputs_embeds`` to ModernBERT and
        never passes ``frame_mask``, so the token embedding table and mask token
        cannot contribute to the loss. Leaving them trainable breaks DDP with
        ``find_unused_parameters=False`` in stages where the gesture encoder is
        otherwise trainable.
        """
        mask_token = getattr(self.gesture_encoder, "mask_token", None)
        if isinstance(mask_token, nn.Parameter):
            mask_token.requires_grad_(False)

        embeddings = getattr(getattr(self.gesture_encoder, "transformer", None), "embeddings", None)
        tok_embeddings = getattr(embeddings, "tok_embeddings", None)
        if tok_embeddings is not None:
            for param in tok_embeddings.parameters():
                param.requires_grad_(False)

    def _load_gesture_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        prefix = "gesture_encoder."
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub:
            sub = state
        missing, unexpected = self.gesture_encoder.load_state_dict(sub, strict=False)
        logger.info(
            "Loaded gesture encoder weights from %s (missing=%d, unexpected=%d)",
            path,
            len(missing),
            len(unexpected),
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._frozen_gesture:
            self.gesture_encoder.eval()
        if self._frozen_t5:
            self.t5.eval()
        return self

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing on T5 and (when trainable) the gesture encoder."""
        kwargs = {"use_reentrant": False}
        self.t5.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        self.t5.config.use_cache = False
        if not self._frozen_gesture:
            transformer = getattr(self.gesture_encoder, "transformer", None)
            if transformer is not None and hasattr(transformer, "gradient_checkpointing_enable"):
                try:
                    transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
                    transformer.config.use_cache = False
                except TypeError:
                    transformer.gradient_checkpointing_enable()

    def _build_encoder_inputs(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = self.t5.get_input_embeddings()(prompt_input_ids)

        gesture = input_features.to(device=prompt_embeds.device)
        attn = feature_attention_mask.to(device=prompt_embeds.device)

        gesture_dtype = next(self.gesture_encoder.parameters()).dtype
        if self._frozen_gesture:
            with torch.no_grad():
                enc = self.gesture_encoder(
                    gesture.to(gesture_dtype), attention_mask=attn
                )
        else:
            enc = self.gesture_encoder(gesture.to(gesture_dtype), attention_mask=attn)

        token_features = enc["token_features"]
        ds_attn_mask = enc["ds_attn_mask"]

        projected = self.feature_projection(token_features.to(prompt_embeds.dtype))

        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)
        attention_mask = torch.cat(
            [prompt_attention_mask.to(device=projected.device), ds_attn_mask], dim=1
        )
        return inputs_embeds, attention_mask

    def forward(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        inputs_embeds, attention_mask = self._build_encoder_inputs(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        for unused in (
            "length",
            "sample_ids",
            "video_paths",
            "dataset_names",
            "languages",
            "target_texts",
            "prompt_texts",
            "num_items_in_batch",
        ):
            kwargs.pop(unused, None)

        if labels is None or not getattr(self, "_fused_ce_enabled", False):
            return self.t5(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        if (
            kwargs.get("decoder_input_ids") is None
            and kwargs.get("decoder_inputs_embeds") is None
        ):
            kwargs["decoder_input_ids"] = self.t5._shift_right(labels)
        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            **kwargs,
        )
        loss = _t5_fused_loss(self.t5, outputs.logits, labels)
        from transformers.modeling_outputs import Seq2SeqLMOutput
        return Seq2SeqLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            decoder_hidden_states=getattr(outputs, "decoder_hidden_states", None),
            decoder_attentions=getattr(outputs, "decoder_attentions", None),
            cross_attentions=getattr(outputs, "cross_attentions", None),
            encoder_last_hidden_state=getattr(outputs, "encoder_last_hidden_state", None),
            encoder_hidden_states=getattr(outputs, "encoder_hidden_states", None),
            encoder_attentions=getattr(outputs, "encoder_attentions", None),
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        inputs_embeds, attention_mask = self._build_encoder_inputs(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        encoder_outputs = self.t5.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
