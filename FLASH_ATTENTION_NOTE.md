# FlashAttention / SDPA Note for `sign_hiera.py`

Context: feature extraction is currently running on all 4 GPUs, so this is a follow-up note for later rather than something to change right now.

## Current behavior

File: [src/mae_pretraining/models/sign_hiera.py](/home/slimelab/Projects/slt/src/mae_pretraining/models/sign_hiera.py)

- Training and inference both already use `torch.nn.functional.scaled_dot_product_attention(...)` when available.
- This means SDPA is active today; the code is not falling back to manual attention unless SDPA is unavailable.

## Important detail

The remaining optimization opportunity is not "enable SDPA", but "make more SDPA calls Flash-compatible".

- Full-attention path:
  - Current tensor layout starts as 5D: `[B, H, 1, N, D]`.
  - The code already squeezes the singleton window dimension to `[B, H, N, D]` before SDPA when `num_windows == 1`.
  - On this machine, that unmasked 4D case is FlashAttention-compatible.
  - If a padding mask is passed, SDPA is still used, but PyTorch falls back to a non-Flash kernel.

- Mask-unit (MU) attention path:
  - Current tensor layout stays 5D: `[B, H, W, L, D]`.
  - The code currently sends this 5D tensor directly to SDPA.
  - On this machine, that shape is not FlashAttention-compatible.
  - A likely optimization is to reshape it to `[B * W, H, L, D]`, run SDPA, then reshape back.

## Why this matters

- Training is already using SDPA.
- Training is probably still slower than it could be because MU attention is likely missing FlashAttention fast kernels.
- So the expected improvement is from a layout change for MU attention, not from simply "turning on SDPA".

## Suggested later follow-up

1. Update MU attention in `MaskUnitAttention.forward()` to flatten the window dimension:
   - `[B, H, W, L, D] -> [B * W, H, L, D]`
   - run SDPA
   - reshape back after attention
2. Re-benchmark training throughput and memory.
3. Check correctness against the current implementation before using it in a long run.
