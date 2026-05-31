#!/usr/bin/env python3
"""Benchmark HAT SRx4 training step speed.

Config: HAT base (embed_dim=180, window_size=16, depths=6x6, gt_size=256, batch=4).
Measures full training step: forward → loss → backward → optimizer.step → EMA update.
Tests: fp32 + AMP, torch.compile on/off (dynamic=False).
"""
import sys
import time
import copy
import torch
import torch.nn as nn

sys.path.insert(0, sys.path[0] + '/..' if sys.path[0].endswith('scripts') else '.')

from hat.archs.hat_arch import HAT

WARMUP = 10
TIMING = 50
GT_SIZE = 256
BATCH = 4
SCALE = 4
LQ_SIZE = GT_SIZE // SCALE  # 64


def build_model():
    return HAT(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16, compress_ratio=3, squeeze_factor=30,
        conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2,
        upscale=4, upsampler='pixelshuffle', resi_connection='1conv',
        img_range=1.0,
    )


def build_ema(net_g):
    net_ema = copy.deepcopy(net_g)
    for p in net_ema.parameters():
        p.requires_grad = False
    return net_ema


@torch.no_grad()
def update_ema(net_g, net_ema, decay=0.999):
    g_params = list(net_g.parameters())
    e_params = list(net_ema.parameters())
    for gp, ep in zip(g_params, e_params):
        ep.data.mul_(decay).add_(gp.data, alpha=1 - decay)


def bench(use_compile, use_amp):
    precision = "AMP" if use_amp else "fp32"
    mode = "compile" if use_compile else "eager"
    header = f"  {mode:>7s}  {precision:>4s}"

    torch.manual_seed(42)
    model = build_model().cuda()
    net_ema = build_ema(model).cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if use_compile:
        print(f"{header}  compiling...", flush=True)
        model = torch.compile(model, dynamic=False)

    torch.manual_seed(1)
    lq = torch.randn(BATCH, 3, LQ_SIZE, LQ_SIZE, device='cuda')
    gt = torch.randn(BATCH, 3, GT_SIZE, GT_SIZE, device='cuda')

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=use_amp):
            output = model(lq)
            loss = criterion(output, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, net_ema)

    # warmup
    for i in range(1, WARMUP + 1):
        if use_compile and i == 1:
            t0 = time.perf_counter()

        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000

        if use_compile and i == 1:
            print(f"    warmup [compiled in {ms/1000:.0f}s]  step {i:>2d}/{WARMUP}: {ms:8.0f} ms", flush=True)
        else:
            print(f"    {'warmup':>28s}  step {i:>2d}/{WARMUP}: {ms:8.0f} ms", flush=True)

    # benchmark
    times = []
    for i in range(1, TIMING + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        times.append(ms)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    del model, net_ema, optimizer, scaler, lq, gt
    torch.cuda.empty_cache()

    first = times[0]
    last = times[-1]
    print(f"    {'bench':>28s}  avg: {avg_ms:7.0f} ms | min: {min_ms:7.0f} ms | max: {max_ms:7.0f} ms | first: {first:7.0f} ms | last: {last:7.0f} ms | peak: {mem_mb:6.0f} MB", flush=True)
    print(flush=True)
    return avg_ms


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", flush=True)
        sys.exit(1)

    torch.set_float32_matmul_precision('high')

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}", flush=True)
    print(f"Config: HAT SRx4, embed_dim=180, window_size=16, depths=6x6, heads=6", flush=True)
    print(f"        gt_size={GT_SIZE}, batch={BATCH}, L1Loss, Adam+EMA", flush=True)
    print(f"        warmup={WARMUP}, timing={TIMING} iters each", flush=True)
    print(flush=True)

    results = {}
    for use_amp in [False, True]:
        for use_compile in [False, True]:
            key = ('fp32' if not use_amp else 'AMP', 'compile' if use_compile else 'eager')
            results[key] = bench(use_compile=use_compile, use_amp=use_amp)

    fp32_eager = results[('fp32', 'eager')]
    fp32_compile = results[('fp32', 'compile')]
    amp_eager = results[('AMP', 'eager')]
    amp_compile = results[('AMP', 'compile')]

    print(f"{'='*72}", flush=True)
    print("Summary (avg ms/step, lower is better):", flush=True)
    print(f"{'':>12} {'eager':>10} {'compile':>10} {'speedup':>10}", flush=True)
    print(f"  {'fp32':>8} {fp32_eager:10.0f} {fp32_compile:10.0f} {fp32_eager/fp32_compile:9.2f}x", flush=True)
    print(f"  {'AMP':>8} {amp_eager:10.0f} {amp_compile:10.0f} {amp_eager/amp_compile:9.2f}x", flush=True)
    print(f"  {'AMP/spd':>8} {fp32_eager/amp_eager:9.2f}x {fp32_compile/amp_compile:9.2f}x", flush=True)


if __name__ == '__main__':
    main()
