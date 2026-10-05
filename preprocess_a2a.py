"""
A2A 数据预处理：将 TIFF tar shards 转换为归一化 float32 numpy tar shards。

转换逻辑（与 ToDtype(scale=True) 等价）：
  - uint8  (RGB/NIR) → / 255.0   → [0, 1]
  - uint16 (SPOT)    → / 65535.0 → [0, 1]
  - float32 (DEM)    → 直接保留

输出格式：每个样本包含 4 个 .npy 文件（rgb.npy, nir.npy, dem.npy, spot.npy），
shape 为 (C, H, W) 的 float32 numpy 数组。

用法：
  python preprocess_a2a.py --input_dir ./datasets_preprocess/FLAIR_ALL_MODALITIES/train \
                           --output_dir ./datasets_preprocess/FLAIR_A2A_NPY/train \
                           --shard_size 350
"""
import argparse
import glob
import os
import io
import json
import numpy as np
import tifffile
import webdataset as wds


def normalize_to_float32(img_np):
    """按 dtype 缩放到 float32 [0, 1]，与 ToDtype(scale=True) 等价"""
    if img_np.dtype == np.uint8:
        return img_np.astype(np.float32) / 255.0
    elif img_np.dtype == np.uint16:
        return img_np.astype(np.float32) / 65535.0
    elif img_np.dtype in (np.float32, np.float64):
        return img_np.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {img_np.dtype}")


def convert_shards(input_dir, output_dir, shard_size):
    os.makedirs(output_dir, exist_ok=True)

    input_shards = sorted(glob.glob(os.path.join(input_dir, "dataset-*.tar")))
    if not input_shards:
        raise FileNotFoundError(f"No tar shards found in {input_dir}")

    print(f"Found {len(input_shards)} input shards in {input_dir}")

    # 读取所有样本并转换
    sink = wds.ShardWriter(
        os.path.join(output_dir, "dataset-%06d.tar"),
        maxcount=shard_size
    )

    total = 0
    for shard_path in input_shards:
        print(f"Processing {os.path.basename(shard_path)}...")
        dataset = wds.WebDataset(shard_path)
        for sample in dataset:
            key = sample["__key__"]
            out_sample = {"__key__": key}

            for modality in ["rgb", "nir", "dem", "spot"]:
                raw_bytes = sample[f"{modality}.tif"]
                img_np = tifffile.imread(io.BytesIO(raw_bytes))
                img_float = normalize_to_float32(img_np)
                # 确保 (C, H, W) 格式
                if img_float.ndim == 2:
                    img_float = img_float[np.newaxis, ...]
                elif img_float.ndim == 3 and img_float.shape[2] < img_float.shape[0]:
                    img_float = img_float.transpose(2, 0, 1)
                out_sample[f"{modality}.npy"] = img_float.tobytes()

            sink.write(out_sample)
            total += 1
            if total % 500 == 0:
                print(f"  Converted {total} samples...")

    sink.close()
    print(f"Done. Total {total} samples written to {output_dir}")

    # 写 metadata.json
    metadata = {"total_samples": total}
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIFF tar shards to normalized float32 numpy tar shards")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing dataset-*.tar (TIFF)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for numpy tar shards")
    parser.add_argument("--shard_size", type=int, default=350, help="Max samples per output shard")
    args = parser.parse_args()

    convert_shards(args.input_dir, args.output_dir, args.shard_size)
