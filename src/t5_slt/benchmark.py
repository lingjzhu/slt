"""Benchmark original vs efficient (patched) SignLanguageT5.

Compares: numerical precision, forward+backward speed, and peak GPU memory.

Usage:
    python -m t5_slt.benchmark [--model google/t5-v1_1-base] [--device cuda]
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mb(x: int | float) -> str:
    return f"{x / 1024**2:.1f} MB" if isinstance(x, int) else f"{x:.1f} MB"


def _make_batch(
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    input_features = torch.randn(batch_size, seq_len, feature_dim, device=device, dtype=dtype)
    feature_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    prompt_ids = torch.randint(0, vocab_size, (batch_size, 4), device=device)
    prompt_mask = torch.ones(batch_size, 4, device=device, dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, 32), device=device)
    labels[:, -8:] = -100
    return dict(
        input_features=input_features,
        feature_attention_mask=feature_mask,
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        labels=labels,
    )


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def benchmark_precision(model_name: str, feature_dim: int, device: torch.device, dtype: torch.dtype):
    from .model import SignLanguageT5

    print("\n" + "=" * 60)
    print("PRECISION BENCHMARK")
    print("=" * 60)

    model_orig = SignLanguageT5(model_name, feature_dim=feature_dim, use_efficient=False).to(device=device, dtype=dtype)
    model_eff = SignLanguageT5(model_name, feature_dim=feature_dim, use_efficient=True).to(device=device, dtype=dtype)
    model_eff.load_state_dict(model_orig.state_dict(), strict=False)

    vocab_size = model_orig.config.vocab_size
    torch.manual_seed(42)
    batch = _make_batch(2, 64, feature_dim, vocab_size, device, dtype)

    # --- Eval mode (logits + loss) ---
    model_orig.eval()
    model_eff.eval()
    with torch.no_grad():
        out_orig = model_orig(**batch)
        out_eff = model_eff(**batch)

    logits_orig = out_orig.logits
    logits_eff = out_eff.logits
    abs_diff = (logits_orig - logits_eff).abs()
    cos_sim = nn.functional.cosine_similarity(
        logits_orig.reshape(1, -1).float(),
        logits_eff.reshape(1, -1).float(),
    ).item()

    print(f"  Logits max  abs diff : {abs_diff.max().item():.6e}")
    print(f"  Logits mean abs diff : {abs_diff.mean().item():.6e}")
    print(f"  Logits cosine sim    : {cos_sim:.10f}")
    print(f"  Loss (original)      : {out_orig.loss.item():.6f}")
    print(f"  Loss (efficient)     : {out_eff.loss.item():.6f}")
    print(f"  Loss abs diff        : {abs(out_orig.loss.item() - out_eff.loss.item()):.6e}")

    # --- Float32 precision check ---
    print("\n  [Float32 reference]")
    model_orig_f32 = model_orig.float()
    model_eff_f32 = model_eff.float()
    batch_f32 = _make_batch(2, 64, feature_dim, vocab_size, device, torch.float32)
    with torch.no_grad():
        out_orig_f32 = model_orig_f32(**batch_f32)
        out_eff_f32 = model_eff_f32(**batch_f32)
    diff_f32 = (out_orig_f32.logits - out_eff_f32.logits).abs()
    print(f"  FP32 logits max diff : {diff_f32.max().item():.6e}")
    print(f"  FP32 logits mean diff: {diff_f32.mean().item():.6e}")
    cos_f32 = nn.functional.cosine_similarity(
        out_orig_f32.logits.reshape(1, -1),
        out_eff_f32.logits.reshape(1, -1),
    ).item()
    print(f"  FP32 cosine sim      : {cos_f32:.10f}")

    del model_orig, model_eff, model_orig_f32, model_eff_f32
    _cleanup()


# ---------------------------------------------------------------------------
# Speed & memory (parametric)
# ---------------------------------------------------------------------------
def _bench_variant(
    label: str,
    model_name: str,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    use_efficient: bool,
    efficient_sdpa: bool = False,
    efficient_flex: bool = False,
    compile_model: bool = False,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> dict:
    from .model import SignLanguageT5

    _cleanup()
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(42)
    model = SignLanguageT5(
        model_name, feature_dim=feature_dim, use_efficient=use_efficient,
        efficient_sdpa=efficient_sdpa, efficient_flex=efficient_flex,
        efficient_compile=compile_model,
    ).to(device=device, dtype=dtype)

    model.train()
    vocab_size = model.config.vocab_size

    # Warmup (extra warmup for compile)
    n_warmup = warmup_iters * (3 if compile_model else 1)
    for _ in range(n_warmup):
        batch = _make_batch(batch_size, seq_len, feature_dim, vocab_size, device, dtype)
        out = model(**batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    fwd_times, bwd_times = [], []
    for _ in range(bench_iters):
        batch = _make_batch(batch_size, seq_len, feature_dim, vocab_size, device, dtype)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model(**batch)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        model.zero_grad(set_to_none=True)

    peak_alloc = torch.cuda.max_memory_allocated(device)

    avg_fwd = sum(fwd_times) / len(fwd_times) * 1000
    avg_bwd = sum(bwd_times) / len(bwd_times) * 1000
    avg_total = avg_fwd + avg_bwd

    print(f"\n  [{label}]")
    print(f"    Forward  avg : {avg_fwd:7.2f} ms")
    print(f"    Backward avg : {avg_bwd:7.2f} ms")
    print(f"    Total    avg : {avg_total:7.2f} ms")
    print(f"    Peak allocated : {_mb(peak_alloc)}")

    result = {
        "fwd_ms": avg_fwd,
        "bwd_ms": avg_bwd,
        "total_ms": avg_total,
        "peak_allocated_mb": peak_alloc / 1024**2,
    }

    del model
    _cleanup()
    return result


def benchmark_speed_memory(
    model_name: str,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4,
    seq_len: int = 256,
    warmup_iters: int = 5,
    bench_iters: int = 20,
):
    print("\n" + "=" * 60)
    print(f"SPEED & MEMORY BENCHMARK  (bs={batch_size}, seq={seq_len}, dtype={dtype})")
    print("=" * 60)

    # (label, use_efficient, sdpa, flex, compile)
    variants = [
        ("ORIGINAL (eager)", False, False, False, False),
        ("TRITON kernels only", True, False, False, False),
        ("TRITON + SDPA", True, True, False, False),
        ("TRITON + FLEX", True, False, True, False),
        ("TRITON + FLEX + compile", True, False, True, True),
    ]

    results = {}
    for label, use_eff, sdpa, flex, do_compile in variants:
        results[label] = _bench_variant(
            label, model_name, feature_dim, device, dtype,
            batch_size, seq_len, use_eff,
            efficient_sdpa=sdpa, efficient_flex=flex, compile_model=do_compile,
            warmup_iters=warmup_iters, bench_iters=bench_iters,
        )

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    orig = results["ORIGINAL (eager)"]
    for label, r in results.items():
        if label == "ORIGINAL (eager)":
            continue
        speedup = orig["total_ms"] / r["total_ms"]
        mem_diff = (1 - r["peak_allocated_mb"] / orig["peak_allocated_mb"]) * 100
        print(f"  {label}:")
        print(f"    Speedup        : {speedup:.2f}x")
        print(f"    Memory savings : {mem_diff:+.1f}%")
        print(f"    Fwd speedup    : {orig['fwd_ms'] / r['fwd_ms']:.2f}x")
        print(f"    Bwd speedup    : {orig['bwd_ms'] / r['bwd_ms']:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def benchmark_generation(
    model_name: str,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4,
    seq_len: int = 256,
    warmup_iters: int = 3,
    bench_iters: int = 10,
):
    from .model import SignLanguageT5

    print("\n" + "=" * 60)
    print(f"GENERATION BENCHMARK  (bs={batch_size}, seq={seq_len})")
    print("=" * 60)

    # Generation uses kv-cache, so flex/sdpa patches fall back to original forward.
    # We still test triton LN + compile.
    gen_variants = [
        ("ORIGINAL", False, False),
        ("TRITON only", True, False),
        ("TRITON + compile", True, True),
    ]
    results = {}
    for label, use_efficient, do_compile in gen_variants:
        _cleanup()

        torch.manual_seed(42)
        model = SignLanguageT5(
            model_name, feature_dim=feature_dim, use_efficient=use_efficient,
            efficient_sdpa=False, efficient_flex=use_efficient,
            efficient_compile=do_compile,
        ).to(device=device, dtype=dtype)
        model.eval()

        vocab_size = model.config.vocab_size
        gen_inputs = dict(
            input_features=torch.randn(batch_size, seq_len, feature_dim, device=device, dtype=dtype),
            feature_attention_mask=torch.ones(batch_size, seq_len, device=device, dtype=torch.long),
            prompt_input_ids=torch.randint(0, vocab_size, (batch_size, 4), device=device),
            prompt_attention_mask=torch.ones(batch_size, 4, device=device, dtype=torch.long),
        )

        for _ in range(warmup_iters):
            with torch.no_grad():
                model.generate(**gen_inputs, max_length=64, num_beams=1)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(bench_iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**gen_inputs, max_length=64, num_beams=1)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

        peak_alloc = torch.cuda.max_memory_allocated(device)
        avg_time = sum(times) / len(times) * 1000

        print(f"\n  [{label}]")
        print(f"    Generate avg   : {avg_time:7.2f} ms")
        print(f"    Peak allocated : {_mb(peak_alloc)}")
        results[label] = {"gen_ms": avg_time, "peak_mb": peak_alloc / 1024**2}

        del model
        _cleanup()

    orig = results["ORIGINAL"]
    print("\n  " + "-" * 40)
    for label, r in results.items():
        if label == "ORIGINAL":
            continue
        print(f"  {label} speedup: {orig['gen_ms'] / r['gen_ms']:.2f}x, mem: {r['peak_mb']:.0f} MB")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark original vs efficient T5")
    parser.add_argument("--model", type=str, default="google/t5-v1_1-base")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Model       : {args.model}")
    print(f"Device      : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Dtype       : {dtype}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Seq length  : {args.seq_len}")
    print(f"PyTorch     : {torch.__version__}")

    benchmark_precision(args.model, args.feature_dim, device, dtype)
    benchmark_speed_memory(
        args.model, args.feature_dim, device, dtype,
        batch_size=args.batch_size, seq_len=args.seq_len,
        warmup_iters=args.warmup, bench_iters=args.iters,
    )
    benchmark_generation(
        args.model, args.feature_dim, device, dtype,
        batch_size=args.batch_size, seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
