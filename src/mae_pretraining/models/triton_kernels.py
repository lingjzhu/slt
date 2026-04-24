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


# ---------------------------------------------------------------------------
# Fused MAE decoder assembly: scatter visible tokens + fill mask tokens.
#
# Replaces:
#     x_dec = torch.zeros(B, N_all, H, W, D)
#     x_dec[mask_bool_expanded] = x.flatten()          # boolean scatter
#     x_dec = ~mask * mask_token + mask * x_dec        # blend
#
# with one forward kernel (and a pure-torch backward using `rank`).
# ---------------------------------------------------------------------------


@triton.jit
def _decoder_assembly_fwd_kernel(
    X_PTR,            # [B, N_vis, HW, D]
    MASK_PTR,         # [B, N_all] bool (as int8)
    RANK_PTR,         # [B, N_all] int32 — rank[b, i] = cumsum(mask, 1)[b, i] - 1
    MTOK_PTR,         # [D]
    OUT_PTR,          # [B, N_all, HW, D]
    B: tl.constexpr,
    N_ALL: tl.constexpr,
    N_VIS: tl.constexpr,
    HW: tl.constexpr,
    D: tl.constexpr,
    sx_b, sx_n, sx_hw, sx_d,
    so_b, so_n, so_hw, so_d,
    sm_b, sm_n,
    sr_b, sr_n,
    BLOCK_D: tl.constexpr,
):
    pid_bi = tl.program_id(0)        # one program per (b, i)
    pid_hw = tl.program_id(1)        # one program per HW slot
    pid_d  = tl.program_id(2)        # D tile

    b = pid_bi // N_ALL
    i = pid_bi % N_ALL

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    m = tl.load(MASK_PTR + b * sm_b + i * sm_n).to(tl.int1)
    rank = tl.load(RANK_PTR + b * sr_b + i * sr_n)

    if m:
        src_ptrs = (
            X_PTR + b * sx_b + rank * sx_n + pid_hw * sx_hw + d_offs * sx_d
        )
        vals = tl.load(src_ptrs, mask=d_mask, other=0.0)
    else:
        vals = tl.load(MTOK_PTR + d_offs, mask=d_mask, other=0.0)

    out_ptrs = (
        OUT_PTR + b * so_b + i * so_n + pid_hw * so_hw + d_offs * so_d
    )
    tl.store(out_ptrs, vals, mask=d_mask)


class _FusedDecoderAssembly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask_bool, mask_token):
        """
        x:          (B, N_vis, H, W, D), dtype fp/bf16
        mask_bool:  (B, N_all) bool
        mask_token: (D,)        dtype matching x

        Returns:
        out:        (B, N_all, H, W, D)
        """
        assert x.dim() == 5 and mask_bool.dim() == 2 and mask_token.dim() == 1
        B, N_vis, H, W, D = x.shape
        B2, N_all = mask_bool.shape
        assert B == B2, (B, B2)
        assert mask_token.shape[0] == D

        x_c = x.contiguous()
        mtok_c = mask_token.contiguous().to(x.dtype)
        mask_i8 = mask_bool.to(torch.int8).contiguous()
        # rank[b, i] = number of True entries in mask[b, :i+1] - 1
        rank = (mask_i8.to(torch.int32).cumsum(dim=1) - 1).contiguous()

        HW = H * W
        x_flat = x_c.view(B, N_vis, HW, D)
        out = torch.empty(B, N_all, HW, D, device=x.device, dtype=x.dtype)

        BLOCK_D = triton.next_power_of_2(D) if D <= 1024 else 1024
        grid = (B * N_all, HW, triton.cdiv(D, BLOCK_D))

        _decoder_assembly_fwd_kernel[grid](
            x_flat, mask_i8, rank, mtok_c, out,
            B, N_all, N_vis, HW, D,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2), x_flat.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            mask_i8.stride(0), mask_i8.stride(1),
            rank.stride(0), rank.stride(1),
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(mask_bool, rank)
        ctx.shapes = (B, N_vis, H, W, D)
        return out.view(B, N_all, H, W, D)

    @staticmethod
    def backward(ctx, grad_out):
        mask_bool, rank = ctx.saved_tensors
        B, N_vis, H, W, D = ctx.shapes
        grad_out = grad_out.contiguous()

        # grad_x[b, j, h, w, d] = grad_out[b, i_j, h, w, d] where i_j is the
        # position of the j-th True in mask[b,:]. Equivalently, gather the
        # rows of grad_out where mask is True, stably ordered.
        # We rely on mask having the same #True per row (enforced by caller).
        grad_x = grad_out[mask_bool].view(B, N_vis, H, W, D)

        # grad_mask_token = sum over all masked-out positions.
        grad_mtok = grad_out[~mask_bool].sum(dim=0).sum(dim=(0, 1))  # sums H, W

        return grad_x, None, grad_mtok


def fused_decoder_assembly(
    x: torch.Tensor, mask_bool: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    return _FusedDecoderAssembly.apply(x, mask_bool, mask_token)
