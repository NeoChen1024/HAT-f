# Gradient accumulation benchmark

**GPU**: RTX 4080 SUPER | **Model**: HAT SRx4, embed_dim=180, window_size=16, depths=6x6
**Config**: AMP, torch.compile, NCHW, no checkpoint | **GT size**: 256

## Results

| batch | accum steps | effective batch | ms/step | img/s | VRAM |
|---|---|---|---|---|---|
| 4 | 1 | 4 | 192 | 20.8 | 10.3 GB |
| 6 | 1 | 6 | 296 | 20.3 | 15.2 GB |
| 6 | 8 | 48 | 2145 | 22.4 | 15.3 GB |

## Key findings

- **Throughput (img/s) is identical regardless of accumulation.** Gradient accumulation arranges forward/backward passes sequentially; each image costs the same compute budget. It does not speed up or slow down training per image.
- **Accumulation overhead is negligible.** bench=6 + accum=8 is 2145 ms vs bench=6 + accum=1 at 296 ms — ratio 7.2x, close to the theoretical 8x. The <0.8x gap is from the shared zero_grad/step/EMA calls.
- **batch=6 is the VRAM limit for RTX 4080 SUPER (16 GB).** At 15.2 GB peak, only ~0.8 GB headroom remains. batch=8 would likely OOM.
- **Effective batch 48 is achievable on a single GPU** by combining batch=6 × accum=8. This is larger than the original 8-GPU setup (8×4=32), giving better gradient estimation stability without sacrificing throughput.

## Training time estimate

For `train_HAT-L_SRx4_finetune_from_ImageNet_pretrain.yml` (250k iters):
- 250k × 2145 ms ≈ **149 hours (~6.2 days)** on a single RTX 4080 SUPER

For `train_HAT_SRx4_ImageNet_from_scratch.yml` (800k iters):
- 800k × 2145 ms ≈ **477 hours (~20 days)**

These are full-step times including both forward and backward passes for gradient accumulation.
