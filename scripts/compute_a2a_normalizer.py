# --*--coding   : utf-8  --*--
# @Function     : 计算 A2A KLVAE 的 latent normalizer (1/std)
#                统计所有模态的 latent 分布，输出 per-channel mean/std/normalizer

import torch
import numpy as np
from diffusers.models import AutoencoderKL
import webdataset as wds
import glob
import os
import io
import tifffile
import argparse
from tqdm import tqdm


def load_klvae(ckpt_path, vae_config):
    """加载 KLVAE checkpoint"""
    vae = AutoencoderKL(**vae_config)
    state_dict = torch.load(ckpt_path, weights_only=False)['state_dict']
    new_state_dict = {}
    for old_key, value in state_dict.items():
        if old_key.startswith("loss_fn."):
            continue
        new_key = old_key.replace("model.", "") if old_key.startswith("model.") else old_key
        new_state_dict[new_key] = value
    vae.load_state_dict(new_state_dict)
    vae.eval()
    vae.cuda()
    return vae


def decode_tiff(raw_bytes):
    """从 WebDataset 的 bytes 解码 TIFF 为 tensor (C, H, W)，自动处理 uint8/uint16"""
    img_np = tifffile.imread(io.BytesIO(raw_bytes))
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    # 与 dataloader 的 ToDtype(float32, scale=True) 保持一致
    if img_np.dtype == np.uint8:
        img_tensor /= 255.0
    elif img_np.dtype == np.uint16:
        img_tensor /= 65535.0
    return img_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_ckpt', type=str,
                        default='./checkpoints/klvae_a2a/last.ckpt')
    parser.add_argument('--data_dir', type=str,
                        default='./datasets_preprocess/FLAIR_ALL_MODALITIES/train')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = use all available samples')
    args = parser.parse_args()

    vae_config = dict(
        in_channels=3, out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2, act_fn='silu', latent_channels=4,
        norm_num_groups=32, sample_size=512, force_upcast=True,
        use_quant_conv=True, use_post_quant_conv=True, mid_block_add_attention=True,
    )

    print(f"Loading KLVAE from: {args.vae_ckpt}")
    vae = load_klvae(args.vae_ckpt, vae_config)

    print(f"Loading data from: {args.data_dir}")
    allshards = glob.glob(os.path.join(args.data_dir, "dataset-*.tar"))
    if not allshards:
        raise FileNotFoundError(f"No tar shards found in {args.data_dir}")
    print(f"Found {len(allshards)} shards")

    dataset = wds.WebDataset(allshards, resampled=False)
    dataset = dataset.rename(rgb='rgb.tif', nir='nir.tif', dem='dem.tif', spot='spot.tif')
    dataset = dataset.to_tuple('rgb', 'nir', 'dem', 'spot', '__key__')
    loader = dataset  # 直接迭代 dataset，不走 WebLoader（避免 batch 合并）

    modality_names = {0: 'rgb', 1: 'nir', 2: 'dem', 3: 'spot'}
    # per-modality latent statistics
    all_latents = {i: [] for i in range(4)}
    count = 0

    print("Encoding samples through KLVAE...")
    for sample in tqdm(loader):
        rgb_bytes, nir_bytes, dem_bytes, spot_bytes, key = sample
        modality_bytes = {0: rgb_bytes, 1: nir_bytes, 2: dem_bytes, 3: spot_bytes}

        for mod_id in range(4):
            try:
                img = decode_tiff(modality_bytes[mod_id])
            except Exception:
                continue
            # Resize to target size
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(args.image_size, args.image_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            # A2A KLVAE trained with float32 [-1, 1] data
            # ToDtype(float32, scale=True) does NOT scale float inputs

            with torch.no_grad():
                latent = vae.encode(img.unsqueeze(0).cuda()).latent_dist.sample()
            all_latents[mod_id].append(latent.cpu())

        count += 1
        if args.max_samples > 0 and count >= args.max_samples:
            break

    print(f"\nProcessed {count} samples")

    # Compute per-modality and global statistics
    print("\n" + "=" * 60)
    print("Per-modality latent statistics:")
    print("=" * 60)

    global_latents = []
    for mod_id in range(4):
        if not all_latents[mod_id]:
            print(f"  {modality_names[mod_id]}: no data")
            continue
        cat = torch.cat(all_latents[mod_id], dim=0)  # (N, 4, H, W)
        per_channel_mean = cat.mean(dim=[0, 2, 3]).tolist()
        per_channel_std = cat.std(dim=[0, 2, 3]).tolist()
        overall_mean = cat.mean().item()
        overall_std = cat.std().item()
        normalizer = 1.0 / overall_std

        print(f"\n  {modality_names[mod_id].upper()}:")
        print(f"    per-channel mean: {[f'{x:.6f}' for x in per_channel_mean]}")
        print(f"    per-channel std:  {[f'{x:.6f}' for x in per_channel_std]}")
        print(f"    overall mean: {overall_mean:.6f}")
        print(f"    overall std:  {overall_std:.6f}")
        print(f"    normalizer (1/std): {normalizer:.5f}")
        global_latents.append(cat)

    # Global statistics across all modalities
    all_cat = torch.cat(global_latents, dim=0)
    print("\n" + "=" * 60)
    print("Global statistics (all modalities combined):")
    print("=" * 60)
    per_channel_mean = all_cat.mean(dim=[0, 2, 3]).tolist()
    per_channel_std = all_cat.std(dim=[0, 2, 3]).tolist()
    overall_mean = all_cat.mean().item()
    overall_std = all_cat.std().item()
    normalizer = 1.0 / overall_std

    print(f"  per-channel mean: {[f'{x:.6f}' for x in per_channel_mean]}")
    print(f"  per-channel std:  {[f'{x:.6f}' for x in per_channel_std]}")
    print(f"  overall mean: {overall_mean:.6f}")
    print(f"  overall std:  {overall_std:.6f}")
    print(f"  normalizer (1/std): {normalizer:.5f}")

    # Per-channel normalizer
    per_channel_normalizer = [1.0 / s for s in per_channel_std]
    print(f"\n  per-channel normalizer: {[f'{x:.5f}' for x in per_channel_normalizer]}")

    print("\n" + "=" * 60)
    print("Config YAML value:")
    print("=" * 60)
    print(f"  target_normalizer: [{', '.join(f'{x:.5f}' for x in per_channel_normalizer)}]")


if __name__ == '__main__':
    main()
