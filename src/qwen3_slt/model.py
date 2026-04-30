from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3ForCausalLM

from gesture_pretraining.models.gesture_encoder import GestureEncoder

logger = logging.getLogger(__name__)


class Qwen3SLT(nn.Module):
    """Qwen3 causal LM conditioned on a pretrained GestureEncoder.

    Sequence layout per sample:
        [<im_start>user\\n] [gesture features] [\\nTranslate {sign_lang} to {lang}<im_end>\\n<im_start>assistant\\n]
        [target tokens] [<im_end>]

    Gesture features are produced by the GestureEncoder (ModernBERT base, 2x temporal
    downsample) and projected into the Qwen3 hidden size by an MLP. They are inserted
    directly as inputs_embeds (not via a placeholder token) so we don't need to resize
    the Qwen3 embedding table.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        *,
        gesture_in_dim: int = 1104,
        gesture_hidden_size: int = 768,
        gesture_num_hidden_layers: int = 22,
        gesture_num_attention_heads: int = 12,
        gesture_intermediate_size: int = 1152,
        gesture_max_position_embeddings: int = 1024,
        gesture_global_attn_every_n_layers: int = 3,
        gesture_local_attention: int = 128,
        gesture_attn_implementation: str = "flash_attention_2",
        gesture_hidden_activation: str = "silu",
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.1,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
        gesture_checkpoint: str | Path | None = None,
        unfreeze_encoder: bool = False,
        unfreeze_projector: bool = True,
        unfreeze_decoder: bool = False,
        apply_liger: bool = True,
    ) -> None:
        super().__init__()

        self.lm = Qwen3ForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.lm.config.use_cache = False
        self.config = self.lm.config

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if self.eos_token_id is None or self.eos_token_id < 0:
            self.eos_token_id = int(self.tokenizer.eos_token_id)

        self.gesture_encoder = GestureEncoder(
            in_dim=gesture_in_dim,
            hidden_size=gesture_hidden_size,
            num_hidden_layers=gesture_num_hidden_layers,
            num_attention_heads=gesture_num_attention_heads,
            intermediate_size=gesture_intermediate_size,
            max_position_embeddings=gesture_max_position_embeddings,
            global_attn_every_n_layers=gesture_global_attn_every_n_layers,
            local_attention=gesture_local_attention,
            attn_implementation=gesture_attn_implementation,
            hidden_activation=gesture_hidden_activation,
        )
        # Liger swaps must run BEFORE checkpoint load: the pretrained encoder was
        # saved with LigerLayerNorm (bias param) + Liger SwiGLU MLP keys.
        if apply_liger:
            self._apply_liger_to_gesture_encoder()
        if gesture_checkpoint:
            self._load_gesture_checkpoint(gesture_checkpoint)
        self._freeze_unused_gesture_parameters()

        d_model = self.config.hidden_size
        proj_hidden = projection_hidden_dim or d_model
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(gesture_hidden_size),
            nn.Linear(gesture_hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(proj_hidden, d_model),
            nn.Dropout(projection_dropout),
        )

        # Pre-tokenize the prompt prefix (constant across samples).
        prefix = "<|im_start|>user\n"
        self._prefix_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        # Pre-cache the suffix token ids per (sign_lang, output_lang) pair.
        self._suffix_cache: dict[tuple[str, str], torch.Tensor] = {}

        self._apply_freeze(unfreeze_encoder, unfreeze_projector, unfreeze_decoder)

        if apply_liger:
            self._apply_liger_to_lm()

        self.to(dtype=torch_dtype)

    # ── Freezing ─────────────────────────────────────────────────────────────

    def _apply_freeze(self, unfreeze_encoder: bool, unfreeze_projector: bool, unfreeze_decoder: bool) -> None:
        for p in self.gesture_encoder.parameters():
            p.requires_grad_(unfreeze_encoder)
        for p in self.feature_projection.parameters():
            p.requires_grad_(unfreeze_projector)
        for p in self.lm.parameters():
            p.requires_grad_(unfreeze_decoder)
        self._frozen_encoder = not unfreeze_encoder
        self._frozen_projector = not unfreeze_projector
        self._frozen_decoder = not unfreeze_decoder

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self._frozen_encoder:
            self.gesture_encoder.eval()
        if self._frozen_projector:
            self.feature_projection.eval()
        if self._frozen_decoder:
            self.lm.eval()
        return self

    def _freeze_unused_gesture_parameters(self) -> None:
        """Drop pretraining-only params unused by SLT forward to satisfy DDP find_unused_parameters=False."""
        mask_token = getattr(self.gesture_encoder, "mask_token", None)
        if isinstance(mask_token, nn.Parameter):
            mask_token.requires_grad_(False)
        embeddings = getattr(getattr(self.gesture_encoder, "transformer", None), "embeddings", None)
        tok_embeddings = getattr(embeddings, "tok_embeddings", None)
        if tok_embeddings is not None:
            for param in tok_embeddings.parameters():
                param.requires_grad_(False)

    def enable_gradient_checkpointing(self) -> None:
        kwargs = {"use_reentrant": False}
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        self.lm.config.use_cache = False
        if not self._frozen_encoder:
            transformer = getattr(self.gesture_encoder, "transformer", None)
            if transformer is not None and hasattr(transformer, "gradient_checkpointing_enable"):
                try:
                    transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
                    transformer.config.use_cache = False
                except TypeError:
                    transformer.gradient_checkpointing_enable()

    # ── Liger ───────────────────────────────────────────────────────────────

    def _apply_liger_to_lm(self) -> None:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3(
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
                model=self.lm,
            )
            logger.info("Applied Liger kernels (rope+rms_norm+swiglu+FLCE) to Qwen3")
        except Exception as exc:
            logger.warning("Liger Qwen3 patch skipped (%s)", exc)

    def _apply_liger_to_gesture_encoder(self) -> None:
        """Swap LayerNorms + ModernBERT MLPs to Liger versions.

        Run BEFORE loading the pretrained checkpoint so the state-dict keys
        (gate_proj/up_proj/down_proj for MLPs, layernorm bias parameters)
        match the pretrained encoder which was saved post-swap.
        """
        try:
            from liger_kernel.transformers import LigerLayerNorm, LigerSwiGLUMLP, LigerGEGLUMLP
            n_ln = _swap_layernorms(self.gesture_encoder, LigerLayerNorm)
            n_mlp, n_skip = _swap_modernbert_mlps(
                self.gesture_encoder, swiglu_cls=LigerSwiGLUMLP, geglu_cls=LigerGEGLUMLP
            )
            logger.info(
                "Applied Liger kernels to gesture encoder: layernorms=%d mlps=%d (skipped=%d)",
                n_ln, n_mlp, n_skip,
            )
        except Exception as exc:
            logger.warning("Liger gesture-encoder patch skipped (%s)", exc)

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def _load_gesture_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            bin_path = path / "pytorch_model.bin"
            if bin_path.is_file():
                path = bin_path
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        prefix = "gesture_encoder."
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub:
            sub = state
        missing, unexpected = self.gesture_encoder.load_state_dict(sub, strict=False)
        logger.info(
            "Loaded gesture encoder weights from %s (missing=%d, unexpected=%d)",
            path, len(missing), len(unexpected),
        )

    # ── Tokenization helpers ────────────────────────────────────────────────

    def _suffix_ids(self, sign_language: str, output_language: str) -> torch.Tensor:
        key = (sign_language, output_language)
        cached = self._suffix_cache.get(key)
        if cached is not None:
            return cached
        text = (
            f"\nTranslate {sign_language} to {output_language}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self._suffix_cache[key] = ids
        return ids

    def _tokenize_targets(self, target_texts: list[str], max_target_length: int) -> tuple[list[torch.Tensor], list[int]]:
        ids_list: list[torch.Tensor] = []
        lengths: list[int] = []
        for text in target_texts:
            ids = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_target_length - 1)["input_ids"]
            ids = list(ids) + [self.eos_token_id]
            t = torch.tensor(ids, dtype=torch.long)
            ids_list.append(t)
            lengths.append(int(t.numel()))
        return ids_list, lengths

    # ── Encoder side ────────────────────────────────────────────────────────

    def _encode_gesture(
        self,
        gesture: torch.Tensor,
        gesture_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gesture_dtype = next(self.gesture_encoder.parameters()).dtype
        if self._frozen_encoder:
            with torch.no_grad():
                enc = self.gesture_encoder(gesture.to(gesture_dtype), attention_mask=gesture_attention_mask)
        else:
            enc = self.gesture_encoder(gesture.to(gesture_dtype), attention_mask=gesture_attention_mask)
        token_features = enc["token_features"]
        ds_attn_mask = enc["ds_attn_mask"]
        embed_dtype = self.lm.get_input_embeddings().weight.dtype
        projected = self.feature_projection(token_features.to(embed_dtype))
        return projected, ds_attn_mask

    # ── Forward (training) ──────────────────────────────────────────────────

    def forward(
        self,
        gesture: torch.Tensor,
        gesture_attention_mask: torch.Tensor,
        sign_languages: list[str],
        output_languages: list[str],
        target_texts: list[str],
        max_target_length: int = 128,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        device = gesture.device
        embed_layer = self.lm.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype

        gesture_embeds, ds_attn_mask = self._encode_gesture(gesture, gesture_attention_mask)
        # gesture_embeds: (B, T_g, D), ds_attn_mask: (B, T_g)
        B, T_g, D = gesture_embeds.shape
        gesture_lengths = ds_attn_mask.sum(dim=1).tolist()  # per-sample real gesture token count

        # Prefix + per-sample suffix + per-sample target token ids
        prefix_ids = self._prefix_ids.to(device)
        prefix_embeds = embed_layer(prefix_ids).to(embed_dtype)  # (L_pre, D)
        L_pre = prefix_embeds.shape[0]

        suffix_id_list = [self._suffix_ids(sl, ol).to(device) for sl, ol in zip(sign_languages, output_languages)]
        suffix_lengths = [int(t.numel()) for t in suffix_id_list]

        target_id_list, target_lengths = self._tokenize_targets(target_texts, max_target_length)
        target_id_list = [t.to(device) for t in target_id_list]

        total_lengths = [L_pre + gesture_lengths[i] + suffix_lengths[i] + target_lengths[i] for i in range(B)]
        max_len = max(total_lengths)

        inputs_embeds = torch.zeros(B, max_len, D, dtype=embed_dtype, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)

        for i in range(B):
            t_g = gesture_lengths[i]
            l_s = suffix_lengths[i]
            l_t = target_lengths[i]
            cursor = 0

            inputs_embeds[i, cursor:cursor + L_pre] = prefix_embeds
            cursor += L_pre

            inputs_embeds[i, cursor:cursor + t_g] = gesture_embeds[i, :t_g]
            cursor += t_g

            suffix_emb = embed_layer(suffix_id_list[i]).to(embed_dtype)
            inputs_embeds[i, cursor:cursor + l_s] = suffix_emb
            cursor += l_s

            target_emb = embed_layer(target_id_list[i]).to(embed_dtype)
            inputs_embeds[i, cursor:cursor + l_t] = target_emb
            labels[i, cursor:cursor + l_t] = target_id_list[i]
            cursor += l_t

            attention_mask[i, :cursor] = 1

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    # ── Generation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        gesture: torch.Tensor,
        gesture_attention_mask: torch.Tensor,
        sign_languages: list[str],
        output_languages: list[str],
        max_new_tokens: int = 128,
        num_beams: int = 1,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        device = gesture.device
        embed_layer = self.lm.get_input_embeddings()
        embed_dtype = embed_layer.weight.dtype

        gesture_embeds, ds_attn_mask = self._encode_gesture(gesture, gesture_attention_mask)
        B, T_g, D = gesture_embeds.shape
        gesture_lengths = ds_attn_mask.sum(dim=1).tolist()

        prefix_ids = self._prefix_ids.to(device)
        prefix_embeds = embed_layer(prefix_ids).to(embed_dtype)
        L_pre = prefix_embeds.shape[0]

        suffix_id_list = [self._suffix_ids(sl, ol).to(device) for sl, ol in zip(sign_languages, output_languages)]
        suffix_embeds_list = [embed_layer(t).to(embed_dtype) for t in suffix_id_list]
        suffix_lengths = [int(t.numel()) for t in suffix_id_list]

        prefix_lengths = [L_pre + gesture_lengths[i] + suffix_lengths[i] for i in range(B)]
        max_len = max(prefix_lengths)

        # Left-pad so the prefix ends at max_len for clean generation.
        inputs_embeds = torch.zeros(B, max_len, D, dtype=embed_dtype, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i in range(B):
            t_g = gesture_lengths[i]
            l_s = suffix_lengths[i]
            offset = max_len - prefix_lengths[i]
            cursor = offset
            inputs_embeds[i, cursor:cursor + L_pre] = prefix_embeds
            cursor += L_pre
            inputs_embeds[i, cursor:cursor + t_g] = gesture_embeds[i, :t_g]
            cursor += t_g
            inputs_embeds[i, cursor:cursor + l_s] = suffix_embeds_list[i]
            attention_mask[i, offset:max_len] = 1

        gen_kwargs = dict(generation_kwargs)
        gen_kwargs.setdefault("pad_token_id", self.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.eos_token_id)
        gen_kwargs.setdefault("do_sample", False)

        # Toggle KV cache for generation only.
        prev_use_cache = self.lm.config.use_cache
        self.lm.config.use_cache = True
        try:
            generated = self.lm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **gen_kwargs,
            )
        finally:
            self.lm.config.use_cache = prev_use_cache
        return generated  # only newly generated tokens when inputs_embeds is supplied


def _swap_layernorms(module: nn.Module, liger_cls: type) -> int:
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
            n += _swap_layernorms(child, liger_cls)
    return n


def _swap_modernbert_mlps(module: nn.Module, *, swiglu_cls: type, geglu_cls: type) -> tuple[int, int]:
    swapped = skipped = 0
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "ModernBertMLP":
            config = child.config
            activation = getattr(config, "hidden_act", getattr(config, "hidden_activation", "gelu"))
            if not hasattr(config, "hidden_act"):
                config.hidden_act = activation
            if getattr(config, "mlp_dropout", 0.0) != 0.0 or getattr(config, "mlp_bias", False):
                skipped += 1
                continue
            if activation in {"silu", "swish"}:
                new = swiglu_cls(config)
            elif activation in {"gelu", "gelu_pytorch_tanh"}:
                new = geglu_cls(config)
            else:
                skipped += 1
                continue
            with torch.no_grad():
                first, second = child.Wi.weight.chunk(2, dim=0)
                new.gate_proj.weight.copy_(first)
                new.up_proj.weight.copy_(second)
                new.down_proj.weight.copy_(child.Wo.weight)
            new.to(child.Wi.weight.device, dtype=child.Wi.weight.dtype)
            setattr(module, name, new)
            swapped += 1
        else:
            sub_s, sub_k = _swap_modernbert_mlps(child, swiglu_cls=swiglu_cls, geglu_cls=geglu_cls)
            swapped += sub_s
            skipped += sub_k
    return swapped, skipped
