#!/usr/bin/env python3
"""Profile HAT SRx4 with torch.compile mode='reduce-overhead' (CUDA graphs).

Config: HAT base (embed_dim=180, window_size=16, depths=6x6, gt_size=256, batch=4).
AMP + torch.compile, NCHW, no checkpointing.
Compares default vs reduce-overhead compile modes.
"""

import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, sys.path[0] + "/.." if sys.path[0].endswith("scripts") else ".")

from hat.archs.hat_arch import HAT

GT_SIZE = 256
BATCH = 4
SCALE = 4
LQ_SIZE = GT_SIZE // SCALE  # 64


def profile_mode(compile_mode, label):
    print(f"\n{'='*60}")
    print(f"Mode: torch.compile(mode='{compile_mode}')")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = HAT(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2,
        upscale=4,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        img_range=1.0,
        use_checkpoint=False,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.L1Loss()

    model = torch.compile(model, dynamic=False, mode=compile_mode)

    lq = torch.randn(BATCH, 3, LQ_SIZE, LQ_SIZE, device="cuda")
    gt = torch.randn(BATCH, 3, GT_SIZE, GT_SIZE, device="cuda")

    # warmup
    print("Warming up (compile)...", flush=True)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            output = model(lq)
            loss = criterion(output, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    print("Warmup done.")

    torch.cuda.reset_peak_memory_stats()

    # profile
    print("Profiling 1 step...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            output = model(lq)
            loss = criterion(output, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak VRAM: {peak_mb:.0f} MB")

    # quick benchmark
    N = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            output = model(lq)
            loss = criterion(output, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) / N * 1000
    print(f"Avg step: {avg_ms:.0f} ms")

    del model, optimizer, scaler, lq, gt
    torch.cuda.empty_cache()
    return avg_ms, peak_mb


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", flush=True)
        sys.exit(1)

    torch.set_float32_matmul_precision("high")

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"Config: HAT SRx4, embed_dim=180, window_size=16, depths=6x6, batch={BATCH}")
    print(f"        AMP, torch.compile")

    default_ms, default_mem = profile_mode("default", "default")
    reduce_ms, reduce_mem = profile_mode("reduce-overhead", "reduce-overhead")

    print(f"\n{'='*60}")
    print(f"Comparison:")
    print(f"  default:          {default_ms:6.0f} ms/step | {default_mem:6.0f} MB")
    print(f"  reduce-overhead:  {reduce_ms:6.0f} ms/step | {reduce_mem:6.0f} MB")
    print(f"  speedup:          {default_ms/reduce_ms:.2f}x")
    print(f"  VRAM delta:       {reduce_mem - default_mem:+.0f} MB")


if __name__ == "__main__":
    main()
