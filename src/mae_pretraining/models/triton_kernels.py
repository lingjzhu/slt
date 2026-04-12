"""
Triton kernels for fused feature extraction operations.

The main kernel fuses the reroll (reshape from flat tokens to spatial layout)
and spatial mean pooling into a single pass, avoiding materialization of the
full [B, T, H, W, C] intermediate tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_spatial_mean_pool_kernel(
    X,    # [B, N, C] flattened tokens where N = T * S
    OUT,  # [B, T, C] output after spatial mean pooling
    B,
    T: tl.constexpr,
    S: tl.constexpr,
    C,
    stride_xb, stride_xn, stride_xc,
    stride_ob, stride_ot, stride_oc,
    BLOCK_C: tl.constexpr,
):
    """Each program computes the spatial mean for one (b, t) position
    over BLOCK_C channels. The spatial mean pools S contiguous tokens
    (the H*W spatial positions for temporal position t) into one vector."""
    bt = tl.program_id(0)
    c_block = tl.program_id(1)

    b = bt // T
    t = bt % T

    c_offs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # Accumulate over S spatial positions in fp32
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    n_base = t * S
    for s in range(S):
        ptrs = X + b * stride_xb + (n_base + s) * stride_xn + c_offs * stride_xc
        vals = tl.load(ptrs, mask=c_mask, other=0.0)
        acc += vals.to(tl.float32)

    acc = acc / S

    # Store in output dtype
    out_ptrs = OUT + b * stride_ob + t * stride_ot + c_offs * stride_oc
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=c_mask)


def fused_spatial_mean_pool(
    x: torch.Tensor, T: int, S: int
) -> torch.Tensor:
    """Fused reshape + spatial mean pooling.

    Takes flattened token tensor [B, N, C] where N = T * S and returns
    [B, T, C] by averaging over the S spatial positions per temporal slot.
    Avoids materializing the intermediate [B, T, S, C] (or [B, T, H, W, C]) tensor.

    Args:
        x: Input tensor of shape [B, N, C], contiguous.
        T: Number of temporal positions.
        S: Number of spatial positions per temporal position (H * W).

    Returns:
        Tensor of shape [B, T, C].
    """
    assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
    B, N, C = x.shape
    assert N == T * S, f"N={N} != T*S={T}*{S}={T*S}"

    x = x.contiguous()
    out = torch.empty(B, T, C, device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (B * T, triton.cdiv(C, BLOCK_C))

    _fused_spatial_mean_pool_kernel[grid](
        x, out,
        B,
        T, S, C,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_C=BLOCK_C,
    )

    return out
