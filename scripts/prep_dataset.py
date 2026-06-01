#!/usr/bin/env python3
"""Prepare a cropped image dataset from raw high-resolution images.

Usage: from a directory of raw HR images, this script:
  1. Crops each image into overlapping sub-images via sliding window
  2. Generates a meta_info.txt file (required by ImageNetPairedDataset)

The output is ready for HAT-f training with `ImageNetPairedDataset`:
  dataroot_gt: <output-dir>
  meta_info_file: <output-dir>/meta_info.txt
"""

import os
import click
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def _scandir_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    paths = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(os.path.join(folder, f))
    return paths


def _size_worker(args):
    path, crop_size = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    return os.path.basename(path), h, w


def _crop_worker(args):
    path, crop_size, step, thresh_size, save_folder, compression = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = os.path.splitext(os.path.basename(path))[0]

    h, w = img.shape[0:2]

    if h < crop_size or w < crop_size:
        return base, -1, h, w  # skipped: too small

    xs = list(range(0, h - crop_size + 1, step))
    if xs and h - (xs[-1] + crop_size) > thresh_size:
        xs.append(h - crop_size)

    ys = list(range(0, w - crop_size + 1, step))
    if ys and w - (ys[-1] + crop_size) > thresh_size:
        ys.append(w - crop_size)

    total = len(xs) * len(ys)
    idx = 0
    for x in xs:
        for y in ys:
            idx += 1
            patch = img[x : x + crop_size, y : y + crop_size]
            patch = np.ascontiguousarray(patch)
            out_name = f"{base}_s{idx:03d}.png"
            cv2.imwrite(
                os.path.join(save_folder, out_name),
                patch,
                [cv2.IMWRITE_PNG_COMPRESSION, compression],
            )

    return base, total, crop_size, crop_size


def _generate_meta_info(image_dir, meta_path):
    from PIL import Image

    with open(meta_path, "w") as f:
        for name in sorted(os.listdir(image_dir)):
            path = os.path.join(image_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                img = Image.open(path)
                f.write(f"{name} ({img.height},{img.width},{len(img.getbands())})\n")
            except Exception:
                continue
    return meta_path


@click.command()
@click.option("--input-dir", "-i", default=None, help="Directory of raw HR source images (required unless --no-crop)")
@click.option("--output-dir", "-o", default=None, help="Directory for cropped sub-images (required unless --dry-run)")
@click.option("--crop-size", "-s", default=480, show_default=True, help="Sub-image crop size (px)")
@click.option("--step", "-p", default=240, show_default=True, help="Sliding window step (px)")
@click.option("--thresh-size", default=240, show_default=True, help="Discard edge patches narrower than this")
@click.option("--workers", "-w", default=8, show_default=True, help="Parallel crop threads")
@click.option("--compression", default=3, show_default=True, help="PNG compression level (0-9)")
@click.option("--no-crop", is_flag=True, help="Skip cropping; only generate meta_info for existing images")
@click.option("--dry-run", is_flag=True, help="Only scan and report image sizes; no cropping or meta_info")
@click.option("--meta-file", "-m", default=None, help="Meta info output path (default: <output-dir>/meta_info.txt)")
def main(input_dir, output_dir, crop_size, step, thresh_size, workers, compression, no_crop, dry_run, meta_file):
    if not dry_run and not output_dir:
        raise click.UsageError("--output-dir is required (unless --dry-run)")

    if dry_run:
        paths = _scandir_images(input_dir)
        if not paths:
            print(f"No images found in {input_dir}")
            return

        tasks = [(p, crop_size) for p in paths]

        small = []
        total = 0
        sz_dist = {}
        with Pool(workers) as pool:
            for name, h, w in tqdm(
                pool.imap_unordered(_size_worker, tasks),
                total=len(tasks), desc="Scanning", unit="img"
            ):
                if h < crop_size or w < crop_size:
                    small.append((name, h, w))
                else:
                    total += 1
                bucket = h // 100 * 100
                sz_dist[bucket] = sz_dist.get(bucket, 0) + 1

        print(f"\nTotal: {len(tasks)} images")
        print(f"Skipped (<{crop_size}px): {len(small)}")
        print(f"Usable (>= {crop_size}px): {total}")
        print(f"\nSize distribution (height):")
        for lo in sorted(sz_dist):
            print(f"  {lo:>5}-{lo+99:<5}px: {sz_dist[lo]:>6}")

        if small:
            print(f"\nImages smaller than {crop_size}px:")
            for name, h, w in sorted(small, key=lambda x: min(x[1], x[2])):
                print(f"  {name:40s} ({h}x{w})")
        return
    if meta_file is None:
        meta_file = os.path.join(output_dir, "meta_info.txt")

    if not no_crop:
        os.makedirs(output_dir, exist_ok=True)

        paths = _scandir_images(input_dir)
        if not paths:
            print(f"ERROR: no images found in {input_dir}")
            return

        print(f"Input:  {len(paths)} source images in {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Crop:   {crop_size}px, step={step}, thresh={thresh_size}, threads={workers}")
        print()

        tasks = [
            (p, crop_size, step, thresh_size, output_dir, compression)
            for p in paths
        ]

        total_patches = 0
        skipped = []
        with Pool(workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_crop_worker, tasks),
                    total=len(tasks),
                    desc="Cropping",
                    unit="img",
                )
            )
            for name, n, h, w in results:
                if n < 0:
                    skipped.append((name, h, w))
                else:
                    total_patches += n

        if skipped:
            print(f"\nWARNING: {len(skipped)} images skipped (smaller than crop_size={crop_size}):")
            for name, h, w in skipped[:10]:
                print(f"  {name}  ({h}x{w})")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")

        print(f"\nDone. {total_patches} patches saved to {output_dir}\n")
    else:
        print(f"Skipping crop. Using existing images in {output_dir}\n")

    meta_file = _generate_meta_info(output_dir, meta_file)
    count = sum(1 for _ in open(meta_file))
    print(f"Meta info: {count} entries written to {meta_file}")
    print()
    print("Ready for training. Add to your YAML:")
    print(f"  dataroot_gt: {output_dir}")
    print(f"  meta_info_file: {meta_file}")


if __name__ == "__main__":
    main()
