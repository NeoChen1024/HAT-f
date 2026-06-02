# HAT-f

Fork of [HAT](https://github.com/XPixelGroup/HAT) (version `0.1.0`), implementing
"Activating More Pixels in Image Super-Resolution Transformer" (CVPR 2023) and
"HAT: Hybrid Attention Transformer for Image Restoration" (arXiv 2023).

This fork adds modern Python/torch compatibility, `torch.compile` acceleration,
Prodigy/Schedule-Free optimizer support, gradient accumulation, and extended
training diagnostics.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install torch torchvision          # must come first
uv pip install -e ./BasicSR-f             # forked submodule, NOT PyPI basicrsr
uv pip install -e .
```

VGG19 pretrained weights (needed for GAN training):

```bash
python3 -c "import torchvision.models as models; models.vgg19(pretrained=True)"
```

## Train

Single GPU with gradient accumulation (replaces 8-GPU DDP):

```bash
python3 hat/train.py -opt training-config/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain.yml
```

`torch.compile` is controlled via YAML block (no code changes needed):

```yaml
compile:
  mode: default           # default | reduce-overhead | max-autotune
  backend: inductor       # inductor | aot_eager | eager
  dynamic: false          # must be false for HAT (static shapes)
```

With compile enabled: ~1.8x speedup, ~40% VRAM reduction.

### Key YAML options

```yaml
train:
  accum_steps: 8                          # gradient accumulation (single GPU)
  optim_g:
    type: ProdigyPlusScheduleFree         # adaptive LR, no scheduler needed
    lr: 1.0
    betas: [0.95, 0.99]
  scheduler: ~                            # not needed with Schedule-Free
```

Enable TensorBoard: `logger.use_tb_logger: true`, then `tensorboard --logdir tb_logger/`.

## Test / inference

```bash
python3 hat/test.py -opt options/test/HAT_SRx4_ImageNet-pretrain.yml
```

Tile mode available for limited VRAM — see `options/test/HAT_tile_example.yml`.

## Dataset preparation

```bash
# Crop HR images into 480×480 sub-images + meta_info
python3 scripts/prep_dataset.py -i raw_images/ -o datasets/mydata/

# Scan and report size distribution only (no cropping)
python3 scripts/prep_dataset.py --dry-run -i raw_images/

# Validation set: generate GT+LR pairs (mod4 crop + bicubic downscale)
python3 scripts/prep_dataset.py --mode paired --scale 4 -i Set5_HR/ -o datasets/Set5/

# Output directly to LMDB (faster IO during training)
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

## Performance (RTX 4080 SUPER, HAT SRx4, gt_size=256, batch=6, FP32)

| | eager | torch.compile |
|---|---|---|
| no checkpoint | ~321 ms | **~157 ms** (~2.0x) |

Full benchmarks: `python3 scripts/bench_train_speed.py -b 6 -a 8 -w 2 -t 15`

AMP removed — HAT is attention-heavy, FP16 produces NaN, BF16 <10% speedup.

## Project structure

| Path | Purpose |
|------|---------|
| `hat/archs/hat_arch.py` | Core HAT model |
| `hat/models/hat_model.py` | Training model (loss, optimizer, validation) |
| `hat/models/realhatgan_model.py` | GAN-based Real-HAT training |
| `hat/models/realhatmse_model.py` | MSE-based Real-HAT training |
| `hat/data/` | Datasets (ImageNetPaired, Real-ESRGAN degradation) |
| `BasicSR-f/` | Forked BasicSR submodule (v1.4.2, Python 3.13 compatible) |
| `scripts/prep_dataset.py` | Dataset preparation (crop, paired, LMDB, size scan) |
| `scripts/batch_infer.py` | Tiled batch inference with torch.compile backends |
| `scripts/bench_train_speed.py` | Training speed benchmark (FP32, DDP, gradient accum) |
| `training-config/` | Training YAML configs |
| `notes/` | Docs: optimizers, torch.compile, gradient accumulation |

## Key changes from upstream

- **Python 3.13 support** with `torch>=2.0` and `einops`
- **`torch.compile` via YAML config** — no manual code changes; `compile:` block controls mode/backend
- **ProdigyPlusScheduleFree optimizer** — adaptive LR with no scheduler, `optimizer.train()/eval()` integration
- **Gradient accumulation** — `accum_steps` YAML param for single-GPU equivalence to multi-GPU batch size
- **Extended TensorBoard logging** — `prodigy_d`, `lr_true`, `grad_norm`, `grad_max`, `weight_norm`, `vram_gb`
- **Resume training** — `--auto_resume` flag reads latest `.state` checkpoint
- **Dataset preparation CLI** — `scripts/prep_dataset.py` with train/paired/LMDB modes
- **Batch inference CLI** — `scripts/batch_infer.py` with tiled processing and selectable compile backends
- **AMP removed** — FP16 produces NaN, BF16 negligible speedup; pure FP32 is simpler

## Known quirks

- Tested on Python 3.13; classifiers only claim 3.7/3.8 but it works.
- `torch.meshgrid` uses `indexing='ij'` explicitly.
- `rgb2ycbcr` import updated from `matlab_functions` → `color_util` (BasicSR 1.3→1.4).
- Setup uses legacy `setup.py` + `pyproject.toml` shim.
- **PyTorch 2.12 `torch.compile` GPU hang**: inductor backend may cause GPU hang
  during batch inference (especially with many tiles). If this occurs, use
  `--compile eager` or `--compile aot_eager` in `scripts/batch_infer.py`.
- **`self.mean` buffer fix for `torch.compile`**: local variable used instead of
  mutating the registered buffer during forward.

## Citations

```
@InProceedings{chen2023activating,
    author    = {Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
    title     = {Activating More Pixels in Image Super-Resolution Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}
@article{chen2023hat,
    title={HAT: Hybrid Attention Transformer for Image Restoration},
    author={Chen, Xiangyu and Wang, Xintao and Zhang, Wenlong and Kong, Xiangtao and Qiao, Yu and Zhou, Jiantao and Dong, Chao},
    journal={arXiv preprint arXiv:2309.05239},
    year={2023}
}
```
