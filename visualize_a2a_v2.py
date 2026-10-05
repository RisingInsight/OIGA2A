# --*--coding   : utf-8  --*--
# @File         : visualize_a2a_v2.py
# @Function     : A2A V2 Stage 1 可视化诊断 - 批量生成三联图 (source | GT | generated)
#                 用于判断生成质量、定位问题，辅助决定是否重训 Stage 1 或进入 Stage 2
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

torch.serialization.add_safe_globals([argparse.Namespace])
from oiga2a_model import OIGA2A
from utils.load_cfg_a2a import load_config
from dataloaders.FLAIR_Dataloader_a2a_v2 import GenericDataModule_V2


MODALITY_NAMES = {0: "RGB", 1: "NIR", 2: "DEM", 3: "SPOT"}


def tensor_to_display(tensor_3ch):
    """Convert a (3, H, W) tensor in [-1, 1] to displayable (H, W, 3) numpy in [0, 1]."""
    arr = tensor_3ch.permute(1, 2, 0).cpu().numpy()
    arr = (arr + 1) / 2
    return np.clip(arr, 0, 1)


def compute_psnr(pred, target):
    """Compute PSNR between two [0,1] numpy arrays."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_l1(pred, target):
    """Compute mean absolute error."""
    return np.mean(np.abs(pred - target))


def create_comparison_figure(source_img, target_img, generated_img,
                              src_name, tgt_name, instruction,
                              psnr_val, l1_val, sample_idx):
    """Create a single comparison figure: source | GT | generated."""
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8], wspace=0.05)

    # Source
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(source_img)
    ax0.set_title(f"Source ({src_name})", fontsize=13, fontweight='bold')
    ax0.axis('off')

    # Ground Truth
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(target_img)
    ax1.set_title(f"GT ({tgt_name})", fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Generated
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(generated_img)
    ax2.set_title(f"Generated ({tgt_name})", fontsize=13, fontweight='bold')
    ax2.axis('off')

    # Info panel
    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')
    info_text = (
        f"Pair: {src_name} → {tgt_name}\n"
        f"Sample: #{sample_idx}\n\n"
        f"PSNR: {psnr_val:.2f} dB\n"
        f"L1: {l1_val:.4f}\n\n"
        f"Instruction:\n{instruction}"
    )
    ax3.text(0.05, 0.5, info_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    return fig


def main():
    parser = argparse.ArgumentParser(description="A2A V2 Visualization Diagnostic")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to generation config YAML')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Override checkpoint path (default: from config)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples per modality pair')
    parser.add_argument('--output_dir', type=str, default='visualize_stage1',
                        help='Output directory for visualizations')
    parser.add_argument('--cfg_scales', type=float, nargs='+', default=None,
                        help='Multiple CFG scales to compare (e.g., 1.0 1.5 2.0 3.0)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for inference (>=2 when using CFG)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = args.ckpt or getattr(cfg.training, 'load', None)
    if not ckpt_path:
        print("Error: No checkpoint path. Use --ckpt <path> or set training.load in config.")
        return
    print(f"Loading model from: {ckpt_path}")

    # Load model (text encoding happens inside forward)
    model = OIGA2A.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.cuda()
    model.eval()

    # Setup dataloader
    modality_pairs = getattr(cfg.datasets, 'modality_pairs', [[0, 1]])

    datamodule = GenericDataModule_V2(
        datasets_folder=cfg.datasets.datasets_folder,
        train_batch_size=1,
        test_batch_size=args.batch_size,
        train_image_size=cfg.training.train_image_size,
        num_workers=cfg.training.num_workers,
        dataset_names=cfg.datasets,
        train_cfg_training=cfg.training,
        mixed_precision=False,
        modality_pairs=modality_pairs,
        instructions_path=getattr(cfg.datasets, 'instructions_path', None),
    )
    datamodule.setup("test")

    # Determine which CFG scales to evaluate (must be > 1.0 for CFG to work)
    raw_scales = args.cfg_scales if args.cfg_scales else [cfg.model.model_config.get('cfg_scale', 1.5)]
    cfg_scales = [s for s in raw_scales if s > 1.0]
    if not cfg_scales:
        print("Warning: No valid CFG scales (>1.0). Using default 1.5")
        cfg_scales = [1.5]
    skipped = [s for s in raw_scales if s <= 1.0]
    if skipped:
        print(f"Note: Skipped CFG scales <= 1.0 (no guidance): {skipped}")
        print(f"Using CFG scales: {cfg_scales}")

    # Collect samples per modality pair
    # Strategy: iterate test data, group by (src_id, tgt_id), stop when enough samples
    samples_per_pair = args.num_samples
    collected = {}  # {(src_id, tgt_id): list of (source, target, src_id, tgt_id, text_tokens)}

    print("Collecting test samples...")
    # Use generate_dataloader (doesn't require trainer attachment)
    if hasattr(cfg, 'generate') and hasattr(cfg.generate, 'dataset_names'):
        gen_datasets = cfg.generate.dataset_names
    else:
        gen_datasets = cfg.datasets.test_datasets
    test_loaders = datamodule.generate_dataloader(gen_datasets)
    if not isinstance(test_loaders, (list, tuple)):
        test_loaders = [test_loaders]
    max_iterations = 500  # safety limit per loader
    iteration = 0
    all_done = False

    for loader in test_loaders:
        for batch in tqdm(loader, desc="Scanning test data"):
            if iteration >= max_iterations:
                break
            iteration += 1

            source, target, src_id, tgt_id, text_tokens = batch
            for i in range(source.shape[0]):
                s_id = int(src_id[i].item())
                t_id = int(tgt_id[i].item())
                pair_key = (s_id, t_id)

                if pair_key not in collected:
                    collected[pair_key] = []

                if len(collected[pair_key]) < samples_per_pair:
                    collected[pair_key].append((
                        source[i:i+1],       # (1, 3, H, W)
                        target[i:i+1],       # (1, 3, H, W)
                        src_id[i:i+1],       # (1,)
                        tgt_id[i:i+1],       # (1,)
                        text_tokens[i:i+1],  # (1, 77)
                    ))

            # Check if all pairs have enough
            all_done = all(
                len(v) >= samples_per_pair
                for v in collected.values()
                if len(v) > 0
            )
            if all_done and len(collected) >= len(modality_pairs):
                break
        if all_done and len(collected) >= len(modality_pairs):
            break

    print(f"\nCollected samples for {len(collected)} modality pairs:")
    for pair_key in sorted(collected.keys()):
        s_name = MODALITY_NAMES.get(pair_key[0], str(pair_key[0]))
        t_name = MODALITY_NAMES.get(pair_key[1], str(pair_key[1]))
        print(f"  {s_name} → {t_name}: {len(collected[pair_key])} samples")

    # Get instruction generator from datamodule for display text
    instruction_gen = getattr(datamodule, 'instruction_generator', None)

    # Run inference and create visualizations
    for cfg_scale in cfg_scales:
        print(f"\n{'='*60}")
        print(f"Running inference with CFG scale = {cfg_scale}")
        print(f"{'='*60}")

        # Override cfg_scale in model (must also update use_cfg flag)
        original_cfg_scale = model.model_config.get('cfg_scale', 1.5)
        model.model_config['cfg_scale'] = cfg_scale
        model.use_cfg = cfg_scale > 1.0

        cfg_dir = os.path.join(args.output_dir, f"cfg_{cfg_scale}")
        os.makedirs(cfg_dir, exist_ok=True)

        for pair_key in sorted(collected.keys()):
            s_id, t_id = pair_key
            src_name = MODALITY_NAMES.get(s_id, str(s_id))
            tgt_name = MODALITY_NAMES.get(t_id, str(t_id))
            pair_name = f"{src_name}2{tgt_name}"

            pair_dir = os.path.join(cfg_dir, pair_name)
            os.makedirs(pair_dir, exist_ok=True)

            print(f"\n  Generating {pair_name}...")

            # forward_with_cfg requires batch_size >= 2 (splits in half internally)
            # Process samples in pairs; duplicate last if odd count
            samples = collected[pair_key]
            if len(samples) % 2 != 0:
                samples = samples + [samples[-1]]  # pad to even
            for sample_idx in range(0, len(samples), 2):
                # Build a batch of 2 (or remaining)
                batch_source = []
                batch_target = []
                batch_s_id = []
                batch_t_id = []
                batch_text = []
                for j in range(sample_idx, min(sample_idx + 2, len(samples))):
                    s, t, si, ti, tt = samples[j]
                    batch_source.append(s)
                    batch_target.append(t)
                    batch_s_id.append(si)
                    batch_t_id.append(ti)
                    batch_text.append(tt)

                source = torch.cat(batch_source, dim=0).cuda()
                target = torch.cat(batch_target, dim=0).cuda()
                s_id_t = torch.cat(batch_s_id, dim=0).cuda()
                t_id_t = torch.cat(batch_t_id, dim=0).cuda()
                text_tokens = torch.cat(batch_text, dim=0).cuda()

                with torch.no_grad():
                    generated = model(source, src_id=s_id_t, tgt_id=t_id_t,
                                     text_tokens=text_tokens, Training=False)

                # Post-process each sample in the batch
                for j in range(generated.shape[0]):
                    idx = sample_idx + j
                    gen_01 = (generated[j] * 0.5 + 0.5).clamp(0, 1)
                    tgt_01 = (target[j] * 0.5 + 0.5).clamp(0, 1)
                    src_01 = (source[j] * 0.5 + 0.5).clamp(0, 1)

                    src_display = src_01.permute(1, 2, 0).cpu().numpy()
                    tgt_display = tgt_01.permute(1, 2, 0).cpu().numpy()
                    gen_display = gen_01.permute(1, 2, 0).cpu().numpy()

                    psnr_val = compute_psnr(gen_display, tgt_display)
                    l1_val = compute_l1(gen_display, tgt_display)

                    if instruction_gen is not None:
                        instruction = instruction_gen.generate(src_name.lower(), tgt_name.lower())
                    else:
                        instruction = f"{src_name} → {tgt_name}"

                    fig = create_comparison_figure(
                        src_display, tgt_display, gen_display,
                        src_name, tgt_name, instruction,
                        psnr_val, l1_val, idx
                    )
                    fig_path = os.path.join(pair_dir, f"sample_{idx}.png")
                    fig.savefig(fig_path, dpi=120, bbox_inches='tight', facecolor='white')
                    plt.close(fig)

                    raw_path = os.path.join(pair_dir, f"sample_{idx}_generated.png")
                    gen_uint8 = (gen_display * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(gen_uint8).save(raw_path)

        # Restore original cfg_scale
        model.model_config['cfg_scale'] = original_cfg_scale
        model.use_cfg = original_cfg_scale > 1.0

    # Create a summary grid: one representative sample per pair
    print(f"\n{'='*60}")
    print("Creating summary grid...")
    print(f"{'='*60}")

    # Summary: create an overview grid using the raw generated images
    for cfg_scale in cfg_scales:
        cfg_dir = os.path.join(args.output_dir, f"cfg_{cfg_scale}")
        summary_path = os.path.join(cfg_dir, "summary_grid.png")

        pair_keys = sorted(collected.keys())
        n_pairs = len(pair_keys)
        if n_pairs == 0:
            continue

        fig, axes = plt.subplots(n_pairs, 1, figsize=(6, 4 * n_pairs))
        if n_pairs == 1:
            axes = [axes]

        for row, pair_key in enumerate(pair_keys):
            s_id, t_id = pair_key
            src_name = MODALITY_NAMES.get(s_id, str(s_id))
            tgt_name = MODALITY_NAMES.get(t_id, str(t_id))
            pair_name = f"{src_name}2{tgt_name}"

            # Show the first generated image
            pair_dir = os.path.join(cfg_dir, pair_name)
            gen_path = os.path.join(pair_dir, "sample_0_generated.png")
            if os.path.exists(gen_path):
                gen_img = np.array(Image.open(gen_path))
                axes[row].imshow(gen_img)
                axes[row].set_title(f"{pair_name} (Generated)", fontsize=12)
            else:
                axes[row].text(0.5, 0.5, "N/A", ha='center', va='center', transform=axes[row].transAxes)
                axes[row].set_title(pair_name, fontsize=12)
            axes[row].axis('off')

        fig.suptitle(f"A2A V2 Stage 1 - CFG={cfg_scale} - Generated Overview", fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"\nDone! Visualizations saved to: {args.output_dir}/")
    print(f"Structure:")
    print(f"  {args.output_dir}/")
    print(f"    cfg_<scale>/")
    print(f"      <SRC>2<TGT>/")
    print(f"        sample_0.png        # Source | GT | Generated comparison")
    print(f"        sample_0_generated.png  # Raw generated image")
    print(f"      summary_grid.png      # Overview of all pairs")


if __name__ == '__main__':
    main()
