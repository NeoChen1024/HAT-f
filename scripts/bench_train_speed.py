#!/usr/bin/env python3
"""Benchmark HAT SRx4 training step speed with torch.compile.

Supports single-GPU, multi-GPU (DDP), and gradient accumulation.
Full training step: forward → loss → backward → optimizer.step → EMA update.
"""
import sys
import os
import time
import copy
import click
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, sys.path[0] + '/..' if sys.path[0].endswith('scripts') else '.')

from hat.archs.hat_arch import HAT

SCALE = 4


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


def setup_ddp(rank, world_size, master_port=29500):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def bench_worker(rank, world_size, batch_size, accum_steps, gt_size, warmup, timing,
                 use_checkpoint, channels_last, master_port, is_master):
    if world_size > 1:
        setup_ddp(rank, world_size, master_port)

    lq_size = gt_size // SCALE
    effective_batch = batch_size * accum_steps * world_size

    torch.manual_seed(42)
    model = build_model(use_checkpoint).cuda(rank)
    net_ema = build_ema(model).cuda(rank)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    model = torch.compile(model, dynamic=False)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    torch.manual_seed(1 + rank)
    lq = torch.randn(batch_size, 3, lq_size, lq_size, device=rank)
    gt = torch.randn(batch_size, 3, gt_size, gt_size, device=rank)
    if channels_last:
        lq = lq.to(memory_format=torch.channels_last)
        gt = gt.to(memory_format=torch.channels_last)

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
    for i in range(1, warmup + 1):
        if i == 1 and is_master:
            print(f"    compiling...", flush=True)

        torch.cuda.synchronize(rank)
        t = time.perf_counter()
        step()
        torch.cuda.synchronize(rank)
        ms = (time.perf_counter() - t) * 1000

        if is_master:
            tag = f"compile {ms/1000:4.0f}s" if i == 1 else ""
            if tag:
                print(f"    {tag}  warmup {i:>2d}/{warmup}: {ms:8.0f} ms", flush=True)
            else:
                print(f"    {'warmup':>28s}  step {i:>2d}/{warmup}: {ms:8.0f} ms", flush=True)

    # benchmark
    times = []
    for i in range(1, timing + 1):
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(rank)
        t = time.perf_counter()
        step()
        torch.cuda.synchronize(rank)
        ms = (time.perf_counter() - t) * 1000
        times.append(ms)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    mem_mb = torch.cuda.max_memory_allocated(rank) / 1024**2

    # gather stats across GPUs
    if world_size > 1:
        avg_t = torch.tensor([avg_ms], device=rank)
        min_t = torch.tensor([min_ms], device=rank)
        max_t = torch.tensor([max_ms], device=rank)
        mem_t = torch.tensor([mem_mb], device=rank)
        dist.reduce(avg_t, 0, op=dist.ReduceOp.MAX)
        dist.reduce(min_t, 0, op=dist.ReduceOp.MAX)
        dist.reduce(max_t, 0, op=dist.ReduceOp.MAX)
        dist.reduce(mem_t, 0, op=dist.ReduceOp.MAX)
        if is_master:
            avg_ms = avg_t.item()
            min_ms = min_t.item()
            max_ms = max_t.item()
            mem_mb = mem_t.item()

    if is_master:
        img_per_sec = effective_batch / (avg_ms / 1000)
        print(f"    {'bench':>28s}  avg: {avg_ms:7.0f} ms | min: {min_ms:7.0f} ms | max: {max_ms:7.0f} ms | peak: {mem_mb:6.0f} MB | {img_per_sec:6.0f} img/s", flush=True)
        print(flush=True)

    if world_size > 1:
        cleanup_ddp()

    del model, net_ema, optimizer, scaler, lq, gt
    torch.cuda.empty_cache()
    return avg_ms


@click.command()
@click.option('--batch-size', '-b', default=6, show_default=True, help='Per-GPU batch size')
@click.option('--accum-steps', '-a', default=8, show_default=True, help='Gradient accumulation steps')
@click.option('--gt-size', '-s', default=256, show_default=True, help='GT crop size (LQ = GT/4)')
@click.option('--warmup', '-w', default=10, show_default=True, help='Warmup iterations')
@click.option('--timing', '-t', default=50, show_default=True, help='Timing iterations')
@click.option('--gpus', '-g', default='0', show_default=True, help='Comma-separated GPU IDs, e.g. 0,1,2,3')
@click.option('--use-checkpoint/--no-checkpoint', default=False, help='Use activation checkpointing')
@click.option('--channels-last/--no-channels-last', default=False, help='Use NHWC memory format')
@click.option('--master-port', default=29500, show_default=True, help='DDP master port')
def main(batch_size, accum_steps, gt_size, warmup, timing, gpus, use_checkpoint, channels_last,
         master_port):
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    torch.set_float32_matmul_precision('high')

    gpu_ids = [int(x.strip()) for x in gpus.split(',')]
    world_size = len(gpu_ids)

    device_name = torch.cuda.get_device_name(gpu_ids[0])
    effective_batch = batch_size * accum_steps * world_size
    fmt = "NHWC" if channels_last else "NCHW"
    ckpt = "yes" if use_checkpoint else "no"

    print(f"GPU: {device_name}", flush=True)
    print(f"GPUs: {gpu_ids} (world_size={world_size})", flush=True)
    print(f"Config: HAT SRx4, embed_dim=180, window_size=16, depths=6x6, heads=6", flush=True)
    print(f"        gt_size={gt_size}, batch={batch_size}, accum={accum_steps}, eff_batch={effective_batch}", flush=True)
    print(f"        AMP, torch.compile, {fmt}, checkpoint={ckpt}", flush=True)
    print(f"        warmup={warmup}, timing={timing} iters each", flush=True)
    print(flush=True)

    if world_size > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
        mp.spawn(
            bench_worker,
            args=(world_size, batch_size, accum_steps, gt_size, warmup, timing,
                  use_checkpoint, channels_last, master_port, False),
            nprocs=world_size,
        )
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        bench_worker(
            0, 1, batch_size, accum_steps, gt_size, warmup, timing,
            use_checkpoint, channels_last, master_port, True,
        )


if __name__ == '__main__':
    main()
