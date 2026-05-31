# HAT-f (Hybrid Attention Transformer fork)

## Project identity

This is a fork of [HAT](https://github.com/XPixelGroup/HAT), renamed **HAT-f** (version `0.1.0`). It implements the paper "Activating More Pixels in Image Super-Resolution Transformer."

## Dependency: BasicSR-f

**Do not install `basicsr` from PyPI.** This project depends on the forked submodule at `./BasicSR-f` (version `1.4.2`), which is a fork of [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) with fixes for modern Python/torch compatibility.

The original `requirements.txt` pinned `basicsr==1.3.4.9`, which does not build on Python 3.13 / uv. It has been relaxed to `basicsr` so the local submodule satisfies the dependency.

## Setup & install (order matters)

```bash
# 1. Create venv
uv venv
source .venv/bin/activate

# 2. Install torch first (build dep for both packages)
uv pip install torch torchvision

# 3. Install BasicSR-f from submodule
uv pip install -e ./BasicSR-f

# 4. Install HAT-f
uv pip install -e .
```

If VGG19 pretrained weights are not cached, download them before running tests/training:

```bash
python3 -c "import torchvision.models as models; models.vgg19(pretrained=True)"
```

Otherwise the first `SRModel` or `PerceptualLoss` init will hang on network I/O.

## API compatibility note: rgb2ycbcr

Between BasicSR 1.3.x and 1.4.2, `rgb2ycbcr` moved from `basicsr.utils.matlab_functions` to `basicsr.utils.color_util`. HAT-f code has been updated accordingly (`hat/data/imagenet_paired_dataset.py`).

## Running tests

```bash
# BasicSR-f unit tests (all 19)
cd BasicSR-f
python3 -m pytest tests/ -v

# Skip data-dependent tests (need test images generated)
python3 -m pytest tests/ -v --ignore=tests/test_data --ignore=tests/test_models
```

Test data must be generated manually from `BasicSR-f/test_scripts/data/baboon.png`:

```bash
cd BasicSR-f
mkdir -p tests/data/gt tests/data/lq
cp test_scripts/data/baboon.png tests/data/gt/
# Downscale 4x for LQ, create second flipped image, write meta_info_gt.txt with 2 entries
```

## Training / testing entrypoints

- Training: `python3 hat/train.py -opt options/train/<config>.yml`
- Inference: `python3 hat/test.py -opt options/test/<config>.yml`

These call into `basicsr.train.train_pipeline` / `basicsr.test.test_pipeline` respectively.

### Training speed benchmark

```bash
python3 scripts/bench_train_speed.py
```

Measures full training step (forward + L1Loss + backward + Adam + EMA) for HAT SRx4,
gt_size=256, batch=4, in fp32 and AMP, with and without `torch.compile`.

## torch.compile

To enable `torch.compile` for training, wrap `self.net_g` in `SRModel.__init__`:

```python
self.net_g = torch.compile(self.net_g, dynamic=False)
```

The `self.mean` buffer mutation in `HAT.forward` was fixed (using a local variable instead
of `self.mean = self.mean.type_as(x)`) — without this fix, `torch.compile` recompiles on
every call due to a CPU→CUDA dispatch key guard failure.

Precision tip: set `torch.set_float32_matmul_precision('high')` before training to allow
TF32 tensor cores, which gives a further ~10% speedup on Ampere/Ada GPUs.

### Benchmark results (RTX 4080 SUPER, HAT SRx4, gt_size=256, batch=4, AMP)

Run: `python3 scripts/bench_train_speed.py`

| | NCHW | NHWC |
|---|---|---|
| no checkpoint | 182 ms | 178 ms |
| checkpoint | 177 ms | 177 ms |

- **`torch.compile` gives ~1.8x speedup** over eager mode (321→182 ms for NCHW no-ckpt).
- **FP32 vs AMP gap is small** (~1.13x) because HAT is attention-heavy (small matmul inner dims,
  lots of memory-bound ops like window partition, softmax, gather) rather than conv/GEMM bound.
- **Activation checkpointing (`use_checkpoint=True`) is slightly faster** (~3%, 182→177 ms).
  HAT's attention intermediate activations (W-MSA scores, OCAB scores) are large; checkpointing
  recomputes them during backward instead of saving/loading them from VRAM. On Ada GPUs the
  FP16 tensor core throughput (~300 TFLOPS) far exceeds memory bandwidth (~736 GB/s), so
  recompute wins over memory fetch.
- **`channels_last` (NHWC) gives negligible benefit** (~2%) because the model is attention-heavy
  rather than conv-heavy. The first compile also takes significantly longer (102s vs 11s) as
  inductor must regenerate all kernels for the new memory format.

## Key architecture files

| File | Purpose |
|------|---------|
| `hat/archs/hat_arch.py` | Core HAT model (Hybrid Attention Transformer) |
| `hat/archs/discriminator_arch.py` | UNet discriminator for GAN training |
| `hat/archs/srvgg_arch.py` | Compact VGG-style SR model |
| `hat/models/hat_model.py` | Standard SR training (extends BasicSR SRModel) |
| `hat/models/realhatgan_model.py` | GAN-based training (extends SRGANModel) |
| `hat/models/realhatmse_model.py` | MSE-based Real-HAT training |
| `hat/data/realesrgan_dataset.py` | Real-ESRGAN-style degradation pipeline |
| `hat/data/imagenet_paired_dataset.py` | Paired image dataset with bicubic downscaling |

## Known quirks

- **Python version**: Tested on 3.13. Classifiers only claim 3.7/3.8 but it works on 3.13.
- **`torch.meshgrid`**: Both HAT-f and BasicSR-f use `indexing='ij'` explicitly (fixed from old implicit behavior).
- **No `pyproject.toml`**: Uses legacy `setup.py` + `setup.cfg`. Build depends on `setuptools`.
- **`setup.py get_version()` at module scope will fail**: The version file is generated by `write_version_py()` inside `if __name__ == '__main__'`, but PEP 517 builds exec setup.py with `__name__` set to `'__main__'`. The `get_version()` function was fixed to use an explicit namespace dict (Python 3 `exec` does not reliably modify `locals()` inside functions).
- **`self.mean` buffer fix for `torch.compile`**: The original `self.mean = self.mean.type_as(x)` in `HAT.forward` reassigns a registered buffer during forward, which caused `torch.compile` recompilation on every call. Fixed by using a local variable (`mean = self.mean.type_as(x)`), see `hat/archs/hat_arch.py:1001`.
