#!/usr/bin/env python3
"""Benchmark HAT SRx4 training step speed with torch.compile and gradient accumulation.

Config: HAT base (embed_dim=180, window_size=16, depths=6x6, gt_size=256).
Measures full training step: forward → loss → backward → optimizer.step → EMA update.
Tests: AMP + torch.compile, NCHW, no checkpoint, varying accum_steps.
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
SCALE = 4
LQ_SIZE = GT_SIZE // SCALE  # 64


def build_model():
    return HAT(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16, compress_ratio=3, squeeze_factor=30,
        conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2,
        upscale=4, upsampler='pixelshuffle', resi_connection='1conv',
        img_range=1.0, use_checkpoint=False,
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


def bench(batch_size, accum_steps):
    effective_batch = batch_size * accum_steps
    header = f"  batch={batch_size}  accum={accum_steps}  eff_batch={effective_batch}"

    torch.manual_seed(42)
    model = build_model().cuda()
    net_ema = build_ema(model).cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    model = torch.compile(model, dynamic=False)

    torch.manual_seed(1)
    lq = torch.randn(batch_size, 3, LQ_SIZE, LQ_SIZE, device='cuda')
    gt = torch.randn(batch_size, 3, GT_SIZE, GT_SIZE, device='cuda')

    def step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            with torch.autocast('cuda'):
                output = model(lq)
                loss = criterion(output, gt) / accum_steps
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, net_ema)

    # warmup
    for i in range(1, WARMUP + 1):
        if i == 1:
            print(f"{header}\n    compiling...", flush=True)

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
    img_per_sec = effective_batch / (avg_ms / 1000)

    torch.cuda.reset_peak_memory_stats()
    del model, net_ema, optimizer, scaler, lq, gt
    torch.cuda.empty_cache()

    print(f"    {'bench':>28s}  avg: {avg_ms:7.0f} ms | min: {min_ms:7.0f} ms | max: {max_ms:7.0f} ms | peak: {mem_mb:6.0f} MB | {img_per_sec:6.0f} img/s", flush=True)
    print(flush=True)
    return avg_ms, img_per_sec, mem_mb


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", flush=True)
        sys.exit(1)

    torch.set_float32_matmul_precision('high')

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}", flush=True)
    print(f"Config: HAT SRx4, embed_dim=180, window_size=16, depths=6x6, heads=6", flush=True)
    print(f"        gt_size={GT_SIZE}, AMP, torch.compile, NCHW, no checkpoint", flush=True)
    print(f"        warmup={WARMUP}, timing={TIMING} iters each", flush=True)
    print(flush=True)

    configs = [
        (4, 1),
        (6, 1),
        (6, 8),
    ]

    results = {}
    for batch_size, accum_steps in configs:
        key = (batch_size, accum_steps)
        results[key] = bench(batch_size, accum_steps)

    # summary
    print(f"{'='*80}", flush=True)
    print("Summary (lower ms = faster step, higher img/s = more throughput):", flush=True)
    print(flush=True)
    print(f"  {'batch':>5} {'accum':>5} {'eff batch':>9} {'ms/step':>10} {'img/s':>10} {'VRAM':>8}", flush=True)
    print(f"  {'-'*5} {'-'*5} {'-'*9} {'-'*10} {'-'*10} {'-'*8}", flush=True)
    for batch_size, accum_steps in configs:
        ms, ips, mem = results[(batch_size, accum_steps)]
        print(f"  {batch_size:5d} {accum_steps:5d} {batch_size*accum_steps:9d} {ms:10.0f} {ips:10.1f} {mem:7.0f} MB", flush=True)


if __name__ == '__main__':
    main()
