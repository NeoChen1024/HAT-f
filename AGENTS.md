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
gt_size=256, batch=6, in FP32 with `torch.compile`. Pure FP32 — AMP removed (negligible
speedup on HAT's attention-heavy architecture, FP16 produces NaN).

## Dataset preparation

```bash
# Crop raw HR images into 480×480 sub-images + meta_info
python3 scripts/prep_dataset.py -i raw_images/ -o datasets/mydata/

# Scan and report size distribution only (no cropping)
python3 scripts/prep_dataset.py --dry-run -i raw_images/

# Validation set: generate GT+LR pairs (mod4 crop + bicubic downscale)
python3 scripts/prep_dataset.py --mode paired --scale 4 -i Set5_HR/ -o datasets/Set5/

# Output directly to LMDB (no intermediate PNGs, faster IO during training)
python3 scripts/prep_dataset.py -i raw_images/ -o datasets/mydata/ --lmdb
```

## Batch inference

```bash
# Default: inductor compile (~2x speedup)
python3 scripts/batch_infer.py -i input/ -o output/

# Choose compile backend (eager = no compile, avoids GPU hang on PyTorch 2.12)
python3 scripts/batch_infer.py -i input/ -o output/ --compile eager
python3 scripts/batch_infer.py -i input/ -o output/ --compile aot_eager
```

Supports `--device auto|cuda|xpu|cpu`, `--model-variant HAT|HAT-S|HAT-L`,
`--tile-size` / `--tile-pad`, `--format png|webp`.

## torch.compile

Configured via YAML `compile:` block in training config (added to `SRModel.__init__`):

```yaml
compile:
  mode: default           # default | reduce-overhead | max-autotune
  backend: inductor       # inductor | aot_eager | eager
  dynamic: false          # must be false for HAT (static shapes)
```

No manual code changes needed. Omit the block entirely to skip compilation.

The `self.mean` buffer mutation in `HAT.forward` was fixed (using a local variable instead
of `self.mean = self.mean.type_as(x)`) — without this fix, `torch.compile` recompiles on
every call due to a CPU→CUDA dispatch key guard failure.

Precision tip: set `torch.set_float32_matmul_precision('high')` before training to allow
TF32 tensor cores, which gives a further ~10% speedup on Ampere/Ada GPUs.

### Benchmark results (RTX 4080 SUPER, HAT SRx4, gt_size=256, batch=6, FP32)

Run: `python3 scripts/bench_train_speed.py -b 6 -a 1 -w 2 -t 15`

| | eager | torch.compile |
|---|---|---|
| no checkpoint | ~321 ms | **~157 ms** (~2.0x) |

- **Activation checkpointing (`use_checkpoint=True`) trades ~24% speed for ~4.5x VRAM reduction**
  (HAT batch=6: 25 GiB → 5.6 GiB, ~3.02s/step → ~3.75s/step).
  HAT's attention intermediate activations (W-MSA scores, OCAB scores) are large; checkpointing
  recomputes them during backward instead of saving/loading them from VRAM.
- **AMP removed**: FP16 produces NaN (Q@K^T and Conv2d overflow 65504), BF16 gives <10% speedup.
  Pure FP32 is simpler and more reliable for HAT.
- **torch.compile reduces VRAM ~40%** via op fusion and memory reuse. Training without
  compile at batch=4 can exceed 20 GiB; with compile it drops to ~11 GiB.
  With use_checkpoint=True (now actually functional), batch=4 uses only 3.4 GiB.
- **PyTorch 2.12 inductor GPU hang**: during batch inference with many tiles, inductor may
  cause GPU hang. Use `--compile eager` or `--compile aot_eager` in `scripts/batch_infer.py`.

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
| `scripts/prep_dataset.py` | Dataset preparation (crop, paired, LMDB, size scan) |
| `scripts/batch_infer.py` | Tiled batch inference with torch.compile backends |
| `scripts/bench_train_speed.py` | Training speed benchmark (FP32, DDP, gradient accum) |
| `training-config/` | Training YAML configs (ProdigyPlusScheduleFree optimizer) |
| `BasicSR-f/basicsr/models/sr_model.py` | torch.compile via YAML `compile:` block |

## Known quirks

- **Python version**: Tested on 3.13. Classifiers only claim 3.7/3.8 but it works on 3.13.
- **`torch.meshgrid`**: Both HAT-f and BasicSR-f use `indexing='ij'` explicitly (fixed from old implicit behavior).
- **No `pyproject.toml`**: Uses legacy `setup.py` + `setup.cfg`. Build depends on `setuptools`.
- **`setup.py get_version()` at module scope will fail**: The version file is generated by `write_version_py()` inside `if __name__ == '__main__'`, but PEP 517 builds exec setup.py with `__name__` set to `'__main__'`. The `get_version()` function was fixed to use an explicit namespace dict (Python 3 `exec` does not reliably modify `locals()` inside functions).
- **`self.mean` buffer fix for `torch.compile`**: The original `self.mean = self.mean.type_as(x)` in `HAT.forward` reassigns a registered buffer during forward, which caused `torch.compile` recompilation on every call. Fixed by using a local variable (`mean = self.mean.type_as(x)`), see `hat/archs/hat_arch.py:1001`.
- **AMP removed**: HAT's attention Q@K^T and Conv2d intermediate values can overflow FP16 (max 65504), producing NaN. BF16 has same exponent range as FP32 but only ~10% speedup. Pure FP32 is simpler and more reliable.
- **PyTorch 2.12 inductor GPU hang**: During batch inference with many tiles, `torch.compile` inductor backend may cause GPU hang. Use `--compile eager` or `--compile aot_eager` in `scripts/batch_infer.py`.
