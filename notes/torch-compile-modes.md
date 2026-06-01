# torch.compile mode comparison for HAT

**GPU**: RTX 4080 SUPER | **Model**: HAT SRx4, embed_dim=180, window_size=16, depths=6x6 | **Config**: batch=4, AMP, no checkpoint, NCHW

## Results

| mode | avg ms/step | compile time | VRAM (PyTorch) | VRAM (nvidia-smi) |
|---|---|---|---|---|
| `default` | **185** | ~100s | 10.2 GB | ~10.5 GB |
| `reduce-overhead` | 186 | ~100s | 10.2 GB | **~16.9 GB** |
| `max-autotune` | 192 | 777s (13 min) | 10.2 GB | ~10.5 GB |

## Interpretation

- **`default` is the best.** 185 ms/step, reasonable compile time, no VRAM penalty.
- **`reduce-overhead` adds zero speedup.** CUDA graphs eliminate CPU dispatch overhead (93ms → 7ms on the backward wrapper), but GPU kernel time dominates wall clock. The CPU dispatch runs concurrently with GPU execution, so saving it on the CPU side does not translate to faster steps. It also locks an extra ~6 GB of VRAM via the CUDA graph memory pool that `torch.cuda.max_memory_allocated()` cannot see (nvidia-smi reveals the real usage).
- **`max-autotune` is 4% slower.** The inductor heuristics in `default` mode already pick near-optimal kernel configurations. Autotuning benchmarks each unique matmul/bmm shape in isolation, but the winner in isolation is not necessarily the winner when hundreds of kernels compete for shared memory and registers in the full forward-backward graph.

## Profiling scripts

- `scripts/bench_train_speed.py` — training step speed benchmark
- `scripts/profile_train_step.py` — single-step profiler (CUDA time + VRAM per op, compares compile modes)
