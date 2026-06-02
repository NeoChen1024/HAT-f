#!/usr/bin/env python3
"""Prepare image datasets for HAT-f training.

Three modes:
  train    Crop HR images into sub-images + generate meta_info (ImageNetPairedDataset)
  paired   Generate GT (mod-cropped) + LR (bicubic) pairs (PairedImageDataset)
  scan     Only scan and report image sizes (--dry-run)
"""

import os
import re
import sys
import shutil
import click
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def _sanitize_name(name):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _modcrop(img, scale):
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


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
    base = _sanitize_name(os.path.splitext(os.path.basename(path))[0])

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
    if save_folder is not None:
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
    else:
        patches = []
        for x in xs:
            for y in ys:
                idx += 1
                patch = img[x : x + crop_size, y : y + crop_size]
                ok, png_bytes = cv2.imencode(".png", patch,
                                              [cv2.IMWRITE_PNG_COMPRESSION, compression])
                key = f"{base}_s{idx:03d}"
                patches.append((key, png_bytes, crop_size, crop_size))
        return base, total, crop_size, crop_size, patches

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
@click.option("--input-dir", "-i", default=None, help="Directory of source images")
@click.option("--output-dir", "-o", default=None, help="Output directory (required unless --dry-run)")
@click.option("--mode", "-M", type=click.Choice(["train", "paired"]), default="train", show_default=True,
              help="train=crop sub-images+meta_info | paired=GT+LR pairs")
@click.option("--scale", "-x", default=4, show_default=True, help="Downscale factor (paired mode)")
@click.option("--crop-size", "-s", default=480, show_default=True, help="Sub-image crop size (train mode)")
@click.option("--step", "-p", default=240, show_default=True, help="Sliding window step (train mode)")
@click.option("--thresh-size", default=240, show_default=True, help="Discard edge patches narrower than this")
@click.option("--workers", "-w", default=8, show_default=True, help="Parallel threads")
@click.option("--compression", default=3, show_default=True, help="PNG compression level (0-9)")
@click.option("--no-crop", is_flag=True, help="Skip cropping; only generate meta_info")
@click.option("--lmdb", is_flag=True, help="Convert output to LMDB format (deletes PNGs afterwards)")
@click.option("--lmdb-batch", default=5000, show_default=True, help="Patches per LMDB transaction (train mode)")
@click.option("--dry-run", is_flag=True, help="Only scan and report image sizes")
@click.option("--meta-file", "-m", default=None, help="Meta info output path (train mode)")
def main(input_dir, output_dir, mode, scale, crop_size, step, thresh_size, workers, compression, no_crop, lmdb, lmdb_batch, dry_run, meta_file):
    if dry_run:
        _scan_mode(input_dir, crop_size, workers)
        return

    if not output_dir:
        raise click.UsageError("--output-dir is required (unless --dry-run)")

    if mode == "paired":
        _paired_mode(input_dir, output_dir, scale, workers, compression)
    else:
        _train_mode(input_dir, output_dir, crop_size, step, thresh_size, workers, compression, no_crop, lmdb, lmdb_batch, meta_file)

    if lmdb and not (mode == "train"):
        _make_lmdb_dirs(output_dir, workers, compression)


def _scan_mode(input_dir, crop_size, workers):
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
            total=len(tasks), desc="Scanning", unit="img", smoothing=0.9,
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


def _paired_worker(args):
    path, scale, gt_dir, lq_dir, compression = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = _sanitize_name(os.path.splitext(os.path.basename(path))[0])

    img = _modcrop(img, scale)
    h, w = img.shape[:2]
    lq = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(gt_dir, f"{base}.png"), img,
                [cv2.IMWRITE_PNG_COMPRESSION, compression])
    cv2.imwrite(os.path.join(lq_dir, f"{base}.png"), lq,
                [cv2.IMWRITE_PNG_COMPRESSION, compression])
    return base, h, w, h // scale, w // scale


def _paired_mode(input_dir, output_dir, scale, workers, compression):
    gt_dir = os.path.join(output_dir, "GTmod4")
    lq_dir = os.path.join(output_dir, f"LRbicx{scale}")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    paths = _scandir_images(input_dir)
    if not paths:
        print(f"ERROR: no images found in {input_dir}")
        return

    print(f"Input:  {len(paths)} source images in {input_dir}")
    print(f"Output: {gt_dir}")
    print(f"        {lq_dir}")
    print(f"Mode:   paired, scale={scale}, mod crop + bicubic downscale")
    print()

    tasks = [(p, scale, gt_dir, lq_dir, compression) for p in paths]
    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_paired_worker, tasks),
            total=len(tasks), desc="Paired", unit="img", smoothing=0.9,
        ):
            pass

    print(f"\nDone. GT saved to {gt_dir}")
    print(f"      LR saved to {lq_dir}")
    print()
    print("Ready for validation. Add to your YAML:")
    print(f"  dataroot_gt: {gt_dir}")
    print(f"  dataroot_lq: {lq_dir}")


def _stream_to_lmdb(results_iter, output_dir, compress_level, batch):
    lmdb_path = output_dir.rstrip("/") + ".lmdb"
    os.makedirs(lmdb_path, exist_ok=True)

    import lmdb

    env = lmdb.open(lmdb_path, map_size=1024**4)  # 1 TB

    total_patches = 0
    skipped = []
    txn = env.begin(write=True)
    meta_path = os.path.join(lmdb_path, "meta_info.txt")
    meta_f = open(meta_path, "w")
    idx = 0

    for result in results_iter:
        if len(result) == 4:
            name, n, h, w = result
            if n < 0:
                skipped.append((name, h, w))
            # no patches (n could be from file-mode, already written to disk)
        else:
            name, n, h, w, patches = result
            if n < 0:
                skipped.append((name, h, w))
                continue
            for key, png_bytes, ph, pw in patches:
                txn.put(key.encode("ascii"), png_bytes.tobytes())
                meta_f.write(f"{key}.png ({ph},{pw},3) {compress_level}\n")
                total_patches += 1
                idx += 1
                if idx % batch == 0:
                    txn.commit()
                    txn = env.begin(write=True)

    txn.commit()
    meta_f.close()
    env.close()

    if skipped:
        print(f"\nWARNING: {len(skipped)} images skipped (smaller than crop_size):")
        for name, h, w in skipped[:10]:
            print(f"  {name}  ({h}x{w})")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    print(f"\nDone. {total_patches} patches saved to {lmdb_path}")
    print(f"Meta info: {total_patches} entries in {meta_path}")
    print()
    print("Ready for training. Add to your YAML:")
    print(f"  dataroot_gt: {lmdb_path}")
    print(f"  io_backend:")
    print(f"    type: lmdb")


def _make_lmdb_dirs(image_dir, workers, compress_level):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BasicSR-f"))
    from basicsr.utils.lmdb_util import make_lmdb_from_imgs

    subdirs = [os.path.join(image_dir, d) for d in sorted(os.listdir(image_dir))
               if os.path.isdir(os.path.join(image_dir, d))]
    if not subdirs:
        subdirs = [image_dir]

    for src_dir in subdirs:
        names = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
        if not names:
            continue
        keys = [os.path.splitext(n)[0] for n in names]
        lmdb_path = src_dir.rstrip("/") + ".lmdb"

        print(f"\nConverting {src_dir} → {lmdb_path}  ({len(names)} images)")
        make_lmdb_from_imgs(
            data_path=src_dir, lmdb_path=lmdb_path,
            img_path_list=names, keys=keys, batch=5000,
            compress_level=compress_level,
        )
        shutil.rmtree(src_dir)
        print(f"Removed {src_dir}")


def _train_mode(input_dir, output_dir, crop_size, step, thresh_size, workers, compression, no_crop, lmdb, lmdb_batch, meta_file):
    if meta_file is None:
        meta_file = os.path.join(output_dir, "meta_info.txt")

    if not no_crop:
        if not lmdb:
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
            (p, crop_size, step, thresh_size, None if lmdb else output_dir, compression)
            for p in paths
        ]

        total_patches = 0
        skipped = []
        with Pool(workers) as pool:
            results = tqdm(
                pool.imap_unordered(_crop_worker, tasks),
                total=len(tasks),
                desc="Cropping",
                unit="img",
                smoothing=0.3,
            )
            if lmdb:
                _stream_to_lmdb(results, output_dir, compression, lmdb_batch)
                return
            else:
                for name, n, h, w in list(results):
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
