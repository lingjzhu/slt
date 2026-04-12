from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


class SignLanguageT5(nn.Module):
    main_input_name = "input_features"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        feature_dim: int,
        projection_dropout: float = 0.1,
        attn_implementation: str = "sdpa",
        use_efficient: bool = False,
        efficient_sdpa: bool = False,
        efficient_flex: bool = True,
        efficient_compile: bool = True,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.attn_implementation = attn_implementation
        self.use_efficient = use_efficient

        config = AutoConfig.from_pretrained(model_name_or_path)
        load_kwargs: dict[str, Any] = {}
        if attn_implementation and not use_efficient:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                config=config,
                **load_kwargs,
            )
        except (TypeError, ValueError) as exc:
            if attn_implementation and (
                "attn_implementation" in str(exc) or "scaled_dot_product_attention" in str(exc)
            ):
                self.t5 = T5ForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    config=config,
                    attn_implementation="eager",
                )
                self.attn_implementation = "eager"
            else:
                raise

        self.config = self.t5.config
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, self.config.d_model),
            nn.Dropout(projection_dropout),
        )

        if use_efficient:
            from .patching import patch_t5_for_efficiency

            stats = patch_t5_for_efficiency(
                self.t5,
                sdpa_attention=efficient_sdpa,
                flex_attention=efficient_flex,
                compile=efficient_compile,
            )
            logger.info("Efficient patches applied: %s", stats)

    def _build_encoder_inputs(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = self.t5.get_input_embeddings()(prompt_input_ids)
        projected_features = self.feature_projection(
            input_features.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        )
        inputs_embeds = torch.cat([prompt_embeds, projected_features], dim=1)
        attention_mask = torch.cat(
            [
                prompt_attention_mask.to(device=projected_features.device),
                feature_attention_mask.to(device=projected_features.device),
            ],
            dim=1,
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
        kwargs.pop("length", None)
        kwargs.pop("sample_ids", None)
        kwargs.pop("video_paths", None)
        kwargs.pop("dataset_names", None)
        kwargs.pop("languages", None)
        kwargs.pop("target_texts", None)
        kwargs.pop("prompt_texts", None)
        kwargs.pop("num_items_in_batch", None)

        if self.use_efficient and labels is not None:
            # Shift labels to create decoder_input_ids (same as T5 internals)
            decoder_input_ids = self.t5._shift_right(labels)
            outputs = self.t5(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=None,
                **kwargs,
            )
            from .kernels import fast_cross_entropy_loss

            outputs.loss = fast_cross_entropy_loss(outputs.logits, labels)
            return outputs

        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
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
