#!/usr/bin/env python3
"""Batch inference for HAT Real-SR x4 with tiled processing.

Input:  directory of images (any format PIL can read)
Output: directory of upscaled images (PNG or WebP, multi-threaded encoding)
Model:  HAT_GAN_Real_SRx4 (Real_HAT_GAN_SRx4.pth)

Tiles are always the same size (tile_size + 2*tile_pad), so
torch.compile(dynamic=False) works without recompilation.
"""

import sys
import os
import math
import time
import click
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, sys.path[0] + "/.." if sys.path[0].endswith("scripts") else ".")

from hat.archs.hat_arch import HAT

WINDOW_SIZE = 16
SCALE = 4


def build_model(variant="HAT"):
    if variant == "HAT-S":
        return HAT(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=144,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            window_size=16,
            compress_ratio=24,
            squeeze_factor=24,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=2,
            upscale=SCALE,
            upsampler="pixelshuffle",
            resi_connection="1conv",
            img_range=1.0,
        )
    if variant == "HAT-L":
        return HAT(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=2,
            upscale=SCALE,
            upsampler="pixelshuffle",
            resi_connection="1conv",
            img_range=1.0,
        )
    # HAT (default)
    return HAT(
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
        upscale=SCALE,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        img_range=1.0,
    )


def extract_state_dict(state, param_key=None):
    if param_key and param_key in state:
        state = state[param_key]
    elif param_key and param_key not in state and "params" in state:
        state = state["params"]
    elif "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def _resolve_device(device_str):
    """Resolve device string to torch.device and a backend tag."""
    if device_str == "cpu":
        return torch.device("cpu"), "cpu"

    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if device_str == "cuda":
        if not has_cuda:
            print("ERROR: CUDA not available", file=sys.stderr)
            sys.exit(1)
        return torch.device("cuda"), "cuda"

    if device_str == "xpu":
        if not has_xpu:
            print("ERROR: XPU not available (install intel-extension-for-pytorch)", file=sys.stderr)
            sys.exit(1)
        return torch.device("xpu"), "xpu"

    # "auto": prefer CUDA, then XPU, then CPU
    if has_cuda:
        return torch.device("cuda"), "cuda"
    if has_xpu:
        return torch.device("xpu"), "xpu"
    return torch.device("cpu"), "cpu"


def _get_device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    if device.type == "xpu":
        return torch.xpu.get_device_name(device.index or 0)
    return "CPU"


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def _set_device_optimizations(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_model(model_path, use_compile, tile_size, tile_pad, device, variant="HAT"):
    model = build_model(variant).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(extract_state_dict(state), strict=True)
    model.eval()
    if use_compile:
        model = torch.compile(model, dynamic=False)
    tile_full = tile_size + 2 * tile_pad
    dry_tile = torch.randn(1, 3, tile_full, tile_full, device=device)
    with torch.inference_mode():
        _ = model(dry_tile)
    _sync_device(device)
    return model


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W), RGB, [0,1]
    return tensor


def pad_to_window(tensor):
    """Pad to multiple of WINDOW_SIZE, falling back when reflect is invalid."""
    _, _, h, w = tensor.shape
    pad_h = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
    pad_w = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE
    if pad_h or pad_w:
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return tensor


def tile_infer(model, lq, tile_size, tile_pad, device):
    """Super-resolve one image via overlapping tile processing."""
    lq = lq.to(device)
    _, _, h, w = lq.shape

    # Pad to window multiple
    lq = pad_to_window(lq)
    _, _, hp, wp = lq.shape

    tile_full = tile_size + 2 * tile_pad
    tiles_x = max(math.ceil(w / tile_size), 1)
    tiles_y = max(math.ceil(h / tile_size), 1)

    # Asymmetric pre-padding so edge tiles can extract full tile_full windows
    last_tx = (tiles_x - 1) * tile_size
    last_ty = (tiles_y - 1) * tile_size
    pad_left = tile_pad
    pad_top = tile_pad
    pad_right = max(0, last_tx + tile_full - wp)
    pad_bottom = max(0, last_ty + tile_full - hp)

    # Use replicate for large pre-padding (reflect fails when pad > input dim)
    lq = F.pad(lq, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    out_h = h * SCALE
    out_w = w * SCALE
    output = torch.zeros(1, 3, out_h + (pad_top + pad_bottom) * SCALE, out_w + (pad_left + pad_right) * SCALE, device=device)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            lx_s = tx * tile_size
            ly_s = ty * tile_size
            lx_e = min(tx * tile_size + tile_size, w)
            ly_e = min(ty * tile_size + tile_size, h)
            tw = lx_e - tx * tile_size
            th = ly_e - ty * tile_size

            tile = lq[:, :, ly_s : ly_s + tile_full, lx_s : lx_s + tile_full].contiguous()

            with torch.inference_mode():
                sr_tile = model(tile)

            out_start = tile_pad * SCALE
            out_end_y = out_start + th * SCALE
            out_end_x = out_start + tw * SCALE

            out_ly = ty * tile_size * SCALE + pad_top * SCALE
            out_lx = tx * tile_size * SCALE + pad_left * SCALE

            output[:, :, out_ly : out_ly + th * SCALE, out_lx : out_lx + tw * SCALE] = sr_tile[
                :, :, out_start:out_end_y, out_start:out_end_x
            ]

    # Remove padding
    output = output[:, :, pad_top * SCALE : pad_top * SCALE + out_h, pad_left * SCALE : pad_left * SCALE + out_w]
    return output.cpu()


def tensor_to_numpy(tensor):
    """(1, 3, H, W) float [0,1] → (H, W, 3) uint8."""
    img = tensor.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    img = img.mul(255).round().clamp(0, 255).byte()
    return img.numpy()


def save_image(arr, path, fmt, quality):
    """Save numpy (H, W, 3) uint8 as PNG or WebP."""
    img = Image.fromarray(arr)
    if fmt == "webp":
        img.save(path, "WEBP", quality=quality)
    else:
        img.save(path, "PNG")


def scan_images(input_dir):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    paths = []
    for f in sorted(os.listdir(input_dir)):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(os.path.join(input_dir, f))
    return paths


def _validate_multiple_of_16(ctx, param, value):
    if value <= 0:
        raise click.BadParameter(f"must be positive, got {value}")
    if value % 16 != 0:
        raise click.BadParameter(f"must be a multiple of 16 (window_size), got {value}")
    return value


def _validate_quality(ctx, param, value):
    if value < 1 or value > 100:
        raise click.BadParameter(f"must be between 1 and 100, got {value}")
    return value


def _validate_workers(ctx, param, value):
    if value <= 0:
        raise click.BadParameter(f"must be positive, got {value}")
    return value


@click.command()
@click.option("--input-dir", "-i", required=True, help="Directory of input images")
@click.option("--output-dir", "-o", required=True, help="Directory for output images")
@click.option(
    "--model-path",
    "-m",
    default="experiments/pretrained_models/Real_HAT_GAN_SRx4.pth",
    show_default=True,
    help="Path to pretrained model",
)
@click.option("--tile-size", default=512, show_default=True, help="Tile size in LR pixels", callback=_validate_multiple_of_16)
@click.option("--tile-pad", default=32, show_default=True, help="Overlap between tiles", callback=_validate_multiple_of_16)
@click.option("--format", "-f", "fmt", type=click.Choice(["png", "webp"]), default="png", show_default=True, help="Output format")
@click.option("--quality", "-q", default=95, show_default=True, help="WebP quality (1-100)", callback=_validate_quality)
@click.option("--compile/--no-compile", "use_compile", default=True, help="Use torch.compile")
@click.option("--workers", "-w", default=4, show_default=True, help="Encoding threads", callback=_validate_workers)
@click.option(
    "--model-variant",
    type=click.Choice(["HAT", "HAT-S", "HAT-L"]),
    default="HAT",
    show_default=True,
    help="Model architecture variant",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["auto", "cuda", "xpu", "cpu"]),
    default="auto",
    show_default=True,
    help="Compute device (auto: CUDA > XPU > CPU)",
)
def main(input_dir, output_dir, model_path, tile_size, tile_pad, fmt, quality, use_compile, workers, model_variant, device):
    torch_device, backend = _resolve_device(device)
    if torch_device.type == "cpu" and device != "cpu":
        print("WARNING: No GPU/XPU detected, falling back to CPU", file=sys.stderr)

    _set_device_optimizations(torch_device)

    os.makedirs(output_dir, exist_ok=True)

    device_name = _get_device_name(torch_device)
    print(f"Model:  {model_path}  ({model_variant})", flush=True)
    print(f"Device: {device_name}  [{backend}]", flush=True)
    print(f"Tile:   {tile_size} + pad {tile_pad}  |  compile: {use_compile}", flush=True)
    print(f"Output: {fmt.upper()}", flush=True)

    print("Loading model...", flush=True)
    model = load_model(model_path, use_compile, tile_size, tile_pad, torch_device, variant=model_variant)
    print("Ready.\n")

    images = scan_images(input_dir)
    print(f"Found {len(images)} images\n")

    futures = []
    max_pending = workers * 2

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, img_path in enumerate(images):
            name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f"{name}.{fmt}")

            t_start = time.perf_counter()

            lq = load_image(img_path)
            _, _, h, w = lq.shape

            sr = tile_infer(model, lq, tile_size, tile_pad, torch_device)

            arr = tensor_to_numpy(sr)
            futures.append(pool.submit(save_image, arr, out_path, fmt, quality))
            if len(futures) >= max_pending:
                futures.pop(0).result()

            elapsed = time.perf_counter() - t_start
            out_h, out_w = h * SCALE, w * SCALE
            print(f"[{idx+1:4d}/{len(images)}] {name}  {w}x{h} → {out_w}x{out_h}  {elapsed:.1f}s  → {out_path}", flush=True)

        # wait for encoding to finish
        for f in as_completed(futures):
            f.result()

    print(f"\n✓ {len(images)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
