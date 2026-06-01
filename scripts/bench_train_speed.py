#!/usr/bin/env python3
"""Benchmark HAT SRx4 training step speed with torch.compile.

Config: HAT base (embed_dim=180, window_size=16, depths=6x6, gt_size=256, batch=4).
Measures full training step: forward → loss → backward → optimizer.step → EMA update.
Tests: AMP + torch.compile, activation checkpointing, channels_last memory format.
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
BATCH = 8
SCALE = 4
LQ_SIZE = GT_SIZE // SCALE  # 64


def build_model(use_checkpoint):
    return HAT(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16, compress_ratio=3, squeeze_factor=30,
        conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2,
        upscale=4, upsampler='pixelshuffle', resi_connection='1conv',
        img_range=1.0, use_checkpoint=use_checkpoint,
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


def bench(use_checkpoint, use_channels_last):
    ckpt_label = "ckpt" if use_checkpoint else "no_ckpt"
    fmt_label = "NHWC" if use_channels_last else "NCHW"
    header = f"  compile  AMP  {ckpt_label:>7s}  {fmt_label}"

    torch.manual_seed(42)
    model = build_model(use_checkpoint).cuda()
    net_ema = build_ema(model).cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    model = torch.compile(model, dynamic=False)

    torch.manual_seed(1)
    lq = torch.randn(BATCH, 3, LQ_SIZE, LQ_SIZE, device='cuda')
    gt = torch.randn(BATCH, 3, GT_SIZE, GT_SIZE, device='cuda')
    if use_channels_last:
        lq = lq.to(memory_format=torch.channels_last)
        gt = gt.to(memory_format=torch.channels_last)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            output = model(lq)
            loss = criterion(output, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, net_ema)

    # warmup (first step triggers compilation)
    for i in range(1, WARMUP + 1):
        if i == 1:
            print(f"{header}  compiling...", flush=True)

        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000

        if i == 1:
            print(f"    compile {ms/1000:4.0f}s  warmup {i:>2d}/{WARMUP}: {ms:8.0f} ms", flush=True)
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

    print(f"    {'bench':>28s}  avg: {avg_ms:7.0f} ms | min: {min_ms:7.0f} ms | max: {max_ms:7.0f} ms | peak: {mem_mb:6.0f} MB", flush=True)
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
    print(f"        gt_size={GT_SIZE}, batch={BATCH}, L1Loss, Adam+EMA, AMP, torch.compile", flush=True)
    print(f"        warmup={WARMUP}, timing={TIMING} iters each", flush=True)
    print(flush=True)

    results = {}
    for use_checkpoint in [False, True]:
        for use_channels_last in [False, True]:
            key = ('ckpt' if use_checkpoint else 'no_ckpt',
                   'NHWC' if use_channels_last else 'NCHW')
            results[key] = bench(use_checkpoint, use_channels_last)

    # summary
    print(f"{'='*72}", flush=True)
    print("Summary (avg ms/step, lower is better) — all AMP + torch.compile:", flush=True)
    print(flush=True)
    print(f"  {'':<15} {'NCHW':>12} {'NHWC':>12}", flush=True)
    print(f"  {'no checkpoint':<15} {results[('no_ckpt','NCHW')]:12.0f} {results[('no_ckpt','NHWC')]:12.0f}", flush=True)
    print(f"  {'checkpoint':<15} {results[('ckpt','NCHW')]:12.0f} {results[('ckpt','NHWC')]:12.0f}", flush=True)
    print(flush=True)

    best_key = min(results, key=results.get)
    print(f"  Best: {best_key[0]} + {best_key[1]}: {results[best_key]:.0f} ms/step", flush=True)


if __name__ == '__main__':
    main()
