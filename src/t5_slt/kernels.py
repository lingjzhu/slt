"""Standalone Triton kernels for efficient T5 inference and training.

Adapted from the Unsloth project (Apache 2.0 License).
Original: https://github.com/unslothai/unsloth
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton version compatibility
# ---------------------------------------------------------------------------
from triton.language.extra import libdevice

triton_tanh = libdevice.tanh
triton_cast = tl.cast

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def calculate_settings(n: int) -> tuple[int, int]:
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@functools.lru_cache(1)
def _is_cdna() -> bool:
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
        return arch in ("gfx940", "gfx941", "gfx942", "gfx950")
    except Exception:
        return False


from contextlib import contextmanager


@contextmanager
def torch_gpu_device(device):
    """Context manager to ensure Triton kernels run on the right device."""
    if device.type == "cuda":
        with torch.cuda.device(device):
            yield
    else:
        yield


# ===================================================================
# Triton RMS LayerNorm  (replaces T5LayerNorm)
# ===================================================================
@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)


def _rms_layernorm_backward_fn(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride
    dX = dY  # in-place

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var
    dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask=mask)


_rms_layernorm_backward = triton.jit(_rms_layernorm_backward_fn)


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float):
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        r = torch.empty(n_rows, dtype=torch.float32, device=device)

        with torch_gpu_device(device):
            _rms_layernorm_forward[(n_rows,)](
                Y, Y.stride(0),
                X, X.stride(0),
                W, W.stride(0),
                r, r.stride(0),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        with torch_gpu_device(dY.device):
            _rms_layernorm_backward[(n_rows,)](
                dY, dY.stride(0),
                dY, dY.stride(0),  # dX written in-place into dY
                X, X.stride(0),
                W, W.stride(0),
                r, r.stride(0),
                n_cols, ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
            )
        return dY.view(*shape), None, None


@torch.compiler.disable
def fast_rms_layernorm(layernorm, X: torch.Tensor) -> torch.Tensor:
    W = layernorm.weight
    eps = (
        layernorm.variance_epsilon
        if hasattr(layernorm, "variance_epsilon")
        else layernorm.eps
    )
    return Fast_RMS_Layernorm.apply(X, W, eps)


# ===================================================================
# Triton Cross Entropy Loss
# ===================================================================
def _cross_entropy_forward_fn(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


_cross_entropy_forward = triton.jit(_cross_entropy_forward_fn)


def _chunked_cross_entropy_forward_fn(
    logits_ptr,
    logits_row_stride: tl.constexpr,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            loss = -1.0 * x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)


_chunked_cross_entropy_forward = triton.jit(_chunked_cross_entropy_forward_fn)


def _cross_entropy_backward_fn(
    logits_ptr,
    logits_row_stride: tl.constexpr,
    dloss_ptr,
    dloss_row_stride: tl.constexpr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


_cross_entropy_backward = triton.jit(_cross_entropy_backward_fn)


class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        n_rows, vocab_size = logits.shape
        device = logits.device
        labels = labels.to(device)

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device=device)

        if n_chunks == 1:
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            if _is_cdna():
                num_warps = num_warps // 2
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device=device)
            with torch_gpu_device(device):
                _cross_entropy_forward[(n_rows,)](
                    logits, logits.stride(0),
                    losses, logsumexp, labels,
                    VOCAB_SIZE=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                )
        else:
            logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=device)
            with torch_gpu_device(device):
                _chunked_cross_entropy_forward[(n_rows, n_chunks)](
                    logits, logits.stride(0),
                    losses, logsumexp, labels,
                    VOCAB_SIZE=vocab_size, N_CHUNKS=n_chunks,
                    BLOCK_SIZE=MAX_FUSED_SIZE,
                    num_warps=32 if not _is_cdna() else 16,
                )
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)

        ctx.save_for_backward(logits, logsumexp, labels)
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        with torch_gpu_device(dlosses.device):
            _cross_entropy_backward[(n_rows, n_blocks)](
                logits, logits.stride(0),
                dlosses, dlosses.stride(0),
                logsumexp, labels,
                VOCAB_SIZE=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
            )
        return logits, None


def fast_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_items: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for cross-entropy loss on (batch, seq_len, vocab) logits."""
    batch, seq_len, d = logits.shape
    assert labels.shape == (batch, seq_len)
    device = logits.device
    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    if torch.is_tensor(n_items):
        n_items = n_items.to(device)
    return loss.sum() / n_items
