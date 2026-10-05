# --*--coding   : utf-8  --*--
# @File         : generate_map_a2a_v2.py
# @Function     : A2A V2 推理/生成入口 - 支持开放指令文本控制
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import tifffile
from PIL import Image
import matplotlib.cm as cm
from matplotlib.colors import LightSource
import clip

torch.serialization.add_safe_globals([argparse.Namespace])
from oiga2a_model import OIGA2A
from utils.load_cfg_a2a import load_config
from dataloaders.FLAIR_Dataloader_a2a_v2 import GenericDataModule_V2


# ================== 可视化工具 ==================
def stretch_image(img, p_min=0.5, p_max=99.5):
    """科学对比度拉伸，输入 (H, W) 或 (H, W, C) 的浮点数组，输出 [0,1]"""
    img_out = np.zeros_like(img)
    if len(img.shape) == 3:
        for i in range(img.shape[2]):
            v = img[:, :, i]
            low, high = np.percentile(v[~np.isnan(v)], (p_min, p_max))
            if high > low:
                img_out[:, :, i] = np.clip((v - low) / (high - low), 0, 1)
            else:
                img_out[:, :, i] = v
    else:
        low, high = np.percentile(img[~np.isnan(img)], (p_min, p_max))
        if high > low:
            img_out = np.clip((img - low) / (high - low), 0, 1)
        else:
            img_out = img
    return img_out


def save_stretched_png(arr_float, save_path):
    """将 [0,1] 浮点数组保存为高质量的 PNG"""
    arr_uint8 = (arr_float * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(save_path)


def visualize_dem(dem_single_channel, save_path):
    """DEM 地形阴影可视化：输入单通道 float (H,W)，值域不定"""
    vmin, vmax = np.percentile(dem_single_channel, 1), np.percentile(dem_single_channel, 99)
    dem_norm = np.clip((dem_single_channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(dem_norm, cmap=cm.terrain, vert_exag=8, blend_mode='soft')
    shaded_rgb = (shaded[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(shaded_rgb).save(save_path)


def visualize_3ch(tensor_c3, save_path):
    """3通道模态可视化：反归一化到 [0,1] 后对比度拉伸"""
    arr = tensor_c3.permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    arr_01 = (arr + 1) / 2
    arr_stretched = stretch_image(arr_01, 0.5, 99.5)
    save_stretched_png(arr_stretched, save_path)


MODALITY_NAMES = {0: "RGB", 1: "NIR", 2: "DEM", 3: "SPOT"}
MODALITY_NAME_TO_ID = {"rgb": 0, "nir": 1, "dem": 2, "spot": 3}


def generate_images_with_instruction(model, datamodule, dataset_names,
                                       instruction=None, output_dir="generate"):
    """A2A V2 推理主逻辑：支持开放指令文本控制"""
    model.eval()
    dataloader_list = datamodule.generate_dataloader(dataset_names)
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize instruction if provided (overrides per-sample instructions)
    if instruction:
        text_tokens = clip.tokenize(instruction, truncate=True).cuda()  # (1, 77)
    else:
        text_tokens = None

    for ds_name, dataloader in zip(dataset_names, dataloader_list):
        print(f"Generating images for dataset: {ds_name}")
        save_dir = os.path.join(output_dir, ds_name)
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            source = batch[0].cuda()       # (B, 3, H, W)
            src_id = batch[2].cuda()        # (B,)
            tgt_id = batch[3].cuda()        # (B,)
            dataloader_tokens = batch[4].cuda()  # (B, 77)

            # Use per-sample tokens from dataloader (open instructions),
            # unless a global instruction override is provided
            if text_tokens is not None:
                batch_text_tokens = text_tokens.expand(source.shape[0], -1)
            else:
                batch_text_tokens = dataloader_tokens

            with torch.no_grad():
                output = model(source, src_id=src_id, tgt_id=tgt_id,
                              text_tokens=batch_text_tokens, Training=False)

            for i in range(output.shape[0]):
                img_tensor = output[i]
                s_id = int(src_id[i].item())
                t_id = int(tgt_id[i].item())
                src_name = MODALITY_NAMES.get(s_id, str(s_id))
                tgt_name = MODALITY_NAMES.get(t_id, str(t_id))
                pair_dir = os.path.join(save_dir, f"{src_name}2{tgt_name}")
                os.makedirs(pair_dir, exist_ok=True)

                # All modalities are 3-channel
                arr = img_tensor.permute(1, 2, 0).cpu().numpy()
                tif_path = os.path.join(pair_dir, f"{batch_idx}_{i}.tif")
                tifffile.imwrite(tif_path, arr)

                # Save visualized PNG
                png_path = os.path.join(pair_dir, f"{batch_idx}_{i}.png")
                visualize_3ch(img_tensor.cpu(), png_path)

        print(f"Finished saving to {save_dir}")


def generate_single_image(model, source_path, src_modality, tgt_modality,
                           instruction, output_path, image_size=(512, 512)):
    """Generate a single image from source with instruction."""
    from torchvision.transforms import v2

    # Load source image
    if source_path.endswith('.tif') or source_path.endswith('.tiff'):
        img_np = tifffile.imread(source_path)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    else:
        from PIL import Image as PILImage
        img = PILImage.open(source_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()

    # Transform
    transform = v2.Compose([
        v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    source = transform(img_tensor).unsqueeze(0).cuda()  # (1, 3, H, W)

    # Get modality IDs
    src_id = torch.tensor([MODALITY_NAME_TO_ID[src_modality.lower()]]).cuda()
    tgt_id = torch.tensor([MODALITY_NAME_TO_ID[tgt_modality.lower()]]).cuda()

    # Tokenize instruction (model encodes internally)
    text_tokens = clip.tokenize(instruction, truncate=True).cuda()  # (1, 77)

    # Generate
    model.eval()
    with torch.no_grad():
        output = model(source, src_id=src_id, tgt_id=tgt_id,
                       text_tokens=text_tokens, Training=False)

    # Save
    img_tensor = output[0]

    # All modalities are 3-channel
    if output_path.endswith('.tif'):
        arr = img_tensor.permute(1, 2, 0).cpu().numpy()
        tifffile.imwrite(output_path, arr)
    else:
        visualize_3ch(img_tensor.cpu(), output_path)

    print(f"Saved output to {output_path}")


def interactive_mode(model, datamodule):
    """Interactive mode: user provides instructions iteratively."""
    print("\n=== A2A V2 Interactive Mode ===")
    print("Type 'quit' to exit, 'source <path>' to set source image")

    current_source = None
    current_src_id = None

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == 'quit':
            break

        if user_input.lower().startswith('source '):
            source_path = user_input[7:].strip()
            if not os.path.exists(source_path):
                print(f"File not found: {source_path}")
                continue
            current_source = source_path
            print(f"Source set to: {source_path}")
            continue

        if user_input.lower().startswith('src_modality '):
            modality = user_input[13:].strip().lower()
            if modality not in MODALITY_NAME_TO_ID:
                print(f"Unknown modality: {modality}. Use: rgb, nir, dem, spot")
                continue
            current_src_id = MODALITY_NAME_TO_ID[modality]
            print(f"Source modality set to: {modality}")
            continue

        if current_source is None:
            print("Please set source image first: source <path>")
            continue

        # Parse instruction: "tgt_modality: instruction"
        if ':' in user_input:
            parts = user_input.split(':', 1)
            tgt_modality = parts[0].strip().lower()
            instruction = parts[1].strip()
        else:
            print("Format: <tgt_modality>: <instruction>")
            print("Example: nir: Convert this aerial photograph to near-infrared vegetation map")
            continue

        if tgt_modality not in MODALITY_NAME_TO_ID:
            print(f"Unknown target modality: {tgt_modality}. Use: rgb, nir, dem, spot")
            continue

        output_path = f"output_{tgt_modality}.png"
        generate_single_image(model, current_source,
                               MODALITY_NAMES.get(current_src_id, "rgb"),
                               tgt_modality, instruction, output_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--instruction', type=str, default=None,
                      help='Text instruction for cross-modal translation')
    args.add_argument('--source_modality', type=str, default=None,
                      help='Source modality (rgb, nir, dem, spot)')
    args.add_argument('--target_modality', type=str, default=None,
                      help='Target modality (rgb, nir, dem, spot)')
    args.add_argument('--source_path', type=str, default=None,
                      help='Path to source image for single image generation')
    args.add_argument('--output_path', type=str, default='output.png',
                      help='Output path for single image generation')
    args.add_argument('--interactive', action='store_true',
                      help='Enable interactive mode')
    args = args.parse_args()

    cfg = load_config(args.config)

    # Load model (contains trained CLIP encoder weights, text encoding happens inside forward)
    ckpt_path = cfg.training.load
    print(f"Loading model from {ckpt_path}")
    model = OIGA2A.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.cuda()
    model.eval()

    if args.interactive:
        # Interactive mode
        modality_pairs = getattr(cfg.datasets, 'modality_pairs', [[0, 1]])
        datamodule = GenericDataModule_V2(
            datasets_folder=cfg.datasets.datasets_folder,
            train_batch_size=1,
            test_batch_size=cfg.training.test_batch_size,
            train_image_size=cfg.training.train_image_size,
            num_workers=cfg.training.num_workers,
            dataset_names=cfg.datasets,
            train_cfg_training=cfg.training,
            mixed_precision=False,
            modality_pairs=modality_pairs,
            instructions_path=getattr(cfg.datasets, 'instructions_path', None),
        )
        datamodule.setup("test")
        interactive_mode(model, datamodule)

    elif args.source_path and args.instruction:
        # Single image generation
        if not args.source_modality or not args.target_modality:
            print("Error: --source_modality and --target_modality required for single image generation")
            exit(1)
        generate_single_image(model, args.source_path,
                               args.source_modality, args.target_modality,
                               args.instruction, args.output_path)

    else:
        # Batch generation
        modality_pairs = getattr(cfg.datasets, 'modality_pairs', [[0, 1]])
        datamodule = GenericDataModule_V2(
            datasets_folder=cfg.datasets.datasets_folder,
            train_batch_size=1,
            test_batch_size=cfg.training.test_batch_size,
            train_image_size=cfg.training.train_image_size,
            num_workers=cfg.training.num_workers,
            dataset_names=cfg.datasets,
            train_cfg_training=cfg.training,
            mixed_precision=False,
            modality_pairs=modality_pairs,
            instructions_path=getattr(cfg.datasets, 'instructions_path', None),
        )
        datamodule.setup("test")

        gen_datasets = cfg.generate.dataset_names
        generate_images_with_instruction(model, datamodule, gen_datasets,
                                          instruction=args.instruction, output_dir="generate")
