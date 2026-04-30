"""Monkey-patch a T5ForConditionalGeneration model with efficient kernels.

Applies three optimizations (all standalone, no external deps beyond torch/triton):

1. **Triton RMS LayerNorm** – optional replacement for every
   ``T5LayerNorm`` forward pass. Disabled by default for training because the
   local Triton backward does not currently produce layernorm weight gradients.
2. **SDPA Attention** – replaces the manual Q*K^T → softmax → V matmul in
   ``T5Attention.forward`` with ``F.scaled_dot_product_attention``, which
   dispatches to memory-efficient attention kernels.  T5's relative position
   bias is passed as an additive ``attn_mask``.
3. **Triton Cross-Entropy Loss** – used at the model-output level in
   ``SignLanguageT5`` (see model.py).
4. **torch.compile** – optionally compiles encoder/decoder stacks for
   additional graph-level fusion.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5LayerNorm,
)

from .kernels import fast_rms_layernorm

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
    _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
except ImportError:
    _HAS_FLEX = False
    _compiled_flex_attention = None
    flex_attention = None
    create_block_mask = None


# ===================================================================
# 1.  Triton RMS LayerNorm patch
# ===================================================================
def _fast_t5_layernorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return fast_rms_layernorm(self, hidden_states)


def _patch_layernorms(model: nn.Module) -> int:
    """Replace every T5LayerNorm.forward with the Triton kernel version."""
    count = 0
    for module in model.modules():
        if isinstance(module, T5LayerNorm):
            module.forward = types.MethodType(_fast_t5_layernorm_forward, module)
            count += 1
    return count


# ===================================================================
# 2.  SDPA Attention patch
# ===================================================================
def _sdpa_t5_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_values=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    """T5Attention.forward rewritten to use F.scaled_dot_product_attention.

    Falls back to the original eager path when ``output_attentions=True``
    or ``layer_head_mask`` is provided.
    """
    if output_attentions or layer_head_mask is not None:
        return self._original_forward(
            hidden_states,
            mask=mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_values=past_key_values,
            layer_head_mask=layer_head_mask,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

    from transformers.cache_utils import EncoderDecoderCache

    batch_size, seq_length = hidden_states.shape[:2]
    is_cross_attention = key_value_states is not None

    query_states = self.q(hidden_states)
    query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    is_updated = False
    if isinstance(past_key_values, EncoderDecoderCache):
        is_updated = past_key_values.is_updated.get(self.layer_idx)
        curr_past_key_value = (
            past_key_values.cross_attention_cache if is_cross_attention
            else past_key_values.self_attention_cache
        )
    else:
        curr_past_key_value = past_key_values

    current_states = key_value_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_values is not None and is_updated:
        key_states = curr_past_key_value.layers[self.layer_idx].keys
        value_states = curr_past_key_value.layers[self.layer_idx].values
    else:
        key_states = self.k(current_states)
        value_states = self.v(current_states)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_values is not None:
            cache_position_arg = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position_arg}
            )
            if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                past_key_values.is_updated[self.layer_idx] = True

    # -- Compute position bias (same logic as eager) --
    if position_bias is None:
        key_length = key_states.shape[-2]
        real_seq_length = query_length if query_length is not None else (
            cache_position[-1] + 1 if cache_position is not None else seq_length
        )
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, key_length),
                device=query_states.device, dtype=query_states.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(
                real_seq_length, key_length,
                device=query_states.device, cache_position=cache_position,
            )
            position_bias = position_bias[:, :, -seq_length:, :]

        if mask is not None:
            causal_mask = mask[:, :, :, :key_states.shape[-2]]
            position_bias = position_bias + causal_mask

    if self.pruned_heads:
        head_mask = torch.ones(position_bias.shape[1], device=position_bias.device)
        head_mask[list(self.pruned_heads)] = 0
        position_bias = position_bias[:, head_mask.bool()]

    # -- SDPA call --
    # position_bias is additive (added to QK^T before softmax), matching SDPA semantics.
    # T5 uses UNSCALED dot-product attention (no 1/sqrt(d_k)), so scale=1.0.
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=position_bias.to(dtype=query_states.dtype),
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
        scale=1.0,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, -1, self.inner_dim)
    attn_output = self.o(attn_output)

    return (attn_output, position_bias)


def _patch_attention(model: nn.Module) -> int:
    """Replace every T5Attention.forward with the SDPA version."""
    count = 0
    for module in model.modules():
        if isinstance(module, T5Attention):
            module._original_forward = module.forward
            module.forward = types.MethodType(_sdpa_t5_attention_forward, module)
            count += 1
    return count


# ===================================================================
# 2b.  FlexAttention patch
# ===================================================================
def _flex_t5_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_values=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    """T5Attention.forward using torch FlexAttention.

    FlexAttention fuses score_mod (here: adding T5 relative position bias) into
    a single CUDA kernel, avoiding materialization of the full QK^T matrix.
    Falls back to the original forward for generation (kv-cache) or any
    unusual path (output_attentions, head mask, pruned heads).
    """
    if (
        output_attentions
        or layer_head_mask is not None
        or past_key_values is not None
        or self.pruned_heads
    ):
        return self._original_forward(
            hidden_states,
            mask=mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_values=past_key_values,
            layer_head_mask=layer_head_mask,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

    batch_size, seq_length = hidden_states.shape[:2]
    is_cross_attention = key_value_states is not None

    query_states = self.q(hidden_states).view(
        batch_size, -1, self.n_heads, self.key_value_proj_dim
    ).transpose(1, 2)

    current_states = key_value_states if is_cross_attention else hidden_states
    key_states = self.k(current_states).view(
        batch_size, -1, self.n_heads, self.key_value_proj_dim
    ).transpose(1, 2)
    value_states = self.v(current_states).view(
        batch_size, -1, self.n_heads, self.key_value_proj_dim
    ).transpose(1, 2)

    key_length = key_states.shape[-2]

    # Compute / reuse position bias (same logic as eager).
    if position_bias is None:
        real_seq_length = query_length if query_length is not None else (
            cache_position[-1] + 1 if cache_position is not None else seq_length
        )
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, key_length),
                device=query_states.device, dtype=query_states.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(
                real_seq_length, key_length,
                device=query_states.device, cache_position=cache_position,
            )
            position_bias = position_bias[:, :, -seq_length:, :]

        if mask is not None:
            causal_mask = mask[:, :, :, :key_length]
            position_bias = position_bias + causal_mask

    # Build full additive bias: (B, H, Q, K).
    bias = position_bias
    if bias.shape[0] == 1 and batch_size != 1:
        bias = bias.expand(batch_size, -1, -1, -1)
    bias = bias.to(dtype=query_states.dtype).contiguous()

    def score_mod(score, b, h, q_idx, k_idx):
        return score + bias[b, h, q_idx, k_idx]

    attn_output = _compiled_flex_attention(
        query_states,
        key_states,
        value_states,
        score_mod=score_mod,
        scale=1.0,
    )

    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, -1, self.inner_dim
    )
    attn_output = self.o(attn_output)

    return (attn_output, position_bias)


def _patch_attention_flex(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, T5Attention):
            module._original_forward = module.forward
            module.forward = types.MethodType(_flex_t5_attention_forward, module)
            count += 1
    return count


# ===================================================================
# 3.  torch.compile patch
# ===================================================================
def _compile_model(model: nn.Module) -> None:
    """Apply torch.compile to encoder and decoder blocks."""
    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        return
    # Compile the main compute-heavy submodules
    model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
    model.decoder = torch.compile(model.decoder, mode="reduce-overhead")


# ===================================================================
# Public API
# ===================================================================
def patch_t5_for_efficiency(
    model: nn.Module,
    *,
    rms_layernorm: bool = False,
    sdpa_attention: bool = False,
    flex_attention: bool = True,
    compile: bool = False,
) -> dict[str, int]:
    """Apply efficiency patches to a T5-based model.

    Args:
        model: T5ForConditionalGeneration (or wrapper containing it).
        sdpa_attention: Replace eager attention with SDPA. Note: T5's position
            bias prevents flash-attention, so SDPA falls back to the
            memory-efficient kernel.  This helps at longer sequences but may
            add overhead for short ones.
        compile: Apply ``torch.compile`` to encoder/decoder stacks.

    Returns a dict with counts of patched modules.
    """
    stats: dict[str, int] = {}
    stats["layernorms"] = _patch_layernorms(model) if rms_layernorm else 0
    if flex_attention:
        if not _HAS_FLEX:
            raise RuntimeError("FlexAttention unavailable; requires PyTorch >= 2.5")
        stats["attention_layers_flex"] = _patch_attention_flex(model)
    elif sdpa_attention:
        stats["attention_layers"] = _patch_attention(model)
    if compile:
        _compile_model(model)
        stats["compiled"] = 1
    return stats


def unpatch_t5(model: nn.Module) -> None:
    """Undo attention patches (restore original forward methods)."""
    for module in model.modules():
        if isinstance(module, T5Attention) and hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
        if isinstance(module, T5LayerNorm) and hasattr(module, "forward"):
            if hasattr(module.forward, "__func__") and module.forward.__func__ is _fast_t5_layernorm_forward:
                del module.forward
