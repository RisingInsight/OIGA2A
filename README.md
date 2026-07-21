# OIGA2A: Open Instruction Guided Any-to-Any Aerial Image Translation via a Single Flow Matching Model

**Anonymous Authors** | Under Review at AAAI 2027

> The code will be released upon paper acceptance.

---

## Abstract

Existing aerial image cross modal translation methods typically train separate models for each source-target modality pair, requiring full retraining for every new modality combination. This results in quadratically increasing total model size and deployment complexity as the number of modalities grows, severely limiting their scalability in practical multi modal remote sensing applications.

To address this limitation, we propose **Open Instruction Guided Any-to-Any (OIGA2A)**, a single weight framework enabling open instruction guided **any-to-any** translation among RGB, Near-Infrared (NIR), Digital Elevation Models (DEM) and SPOT satellite imagery. It leverages a **flow matching** diffusion Transformer operating in a shared latent space learned by a single jointly trained KL-regularized Variational Autoencoder (VAE). A dynamic loss mask router adaptively applies L1+perceptual loss for optical texture modalities and spatial gradient consistency loss for topographic DEM, ensuring faithful reconstruction while preserving unique modality characteristics.

A conditioning scheme integrates open instruction semantics via a LoRA finetuned CLIP encoder and structural alignment via AdaLN-Zero modulation, with classifier free guidance further enhancing inference controllability. Extensive experiments across all 12 possible translation directions demonstrate that our **single unified model** outperforms pair-specific baselines on average across most quantitative metrics, while consolidating all directions into a single deployable weight set and qualitatively demonstrating faithful open-vocabulary instruction following. By consolidating all mappings into one compact model, OIGA2A significantly reduces deployment complexity and storage footprint, enabling efficient deployment for real world remote sensing applications.

---

## Motivation

![Qualitative comparison](figures/introduction.jpg)

*Qualitative comparison between traditional image translation methods and our method.*

The primary contributions are:

1. We propose **OIGA2A**, the first open instruction guided any-to-any translation framework for remote sensing, condensing all 12 bidirectional cross-platform pathways into a single unified weight set and eliminating the quadratic model growth of pairwise methods.

2. We design a unified architecture that tightly couples three components into a single coherent design:
   - **Dynamic Loss Mask Router** — within a single shared KL-VAE, which adaptively routes optical textural losses and geometric gradient consistency constraints per modality, eliminating cross-modal gradient interference while simultaneously preserving photometric fidelity and topographic structure.
   - **Modality Calibrated Flow Matching** — which augments a shared Scalable Interpolant Transformer (SiT) backbone with per-modality latent normalization and lightweight per-target-modality residual adapters, allowing a single flow matching backbone to accommodate the heterogeneous latent statistics of all modalities without sacrificing weight sharing or inference efficiency.
   - **Conditioning Strategy** — that integrates a LoRA-adapted CLIP instruction encoder with AdaLN-Zero modulation jointly conditioned on source modality, target modality, and timestep, enabling simultaneous satisfaction of open expert-level semantic intent and physical modality constraints.

3. Extensive evaluations on all 12 translation permutations of the FLAIR-HUB benchmark show that a single OIGA2A model consistently outperforms seven pairwise trained baselines, reducing average FID by **14.7%** relative to the strongest baseline (Pix2PixHD) while achieving the best average LPIPS and SSIM, using **3x–40x** fewer total parameters than five of the seven pairwise baseline sets (Pix2PixHD, FCDiffusion, CDTSDE, LBM, Dual-App-Bridge); against the two lightweight GAN baselines, OIGA2A's single model is comparable in parameter count, though it still collapses 6–12 separately managed checkpoints into one.

---

## Method

![Overview of the OIGA2A framework](figures/overview.jpg)

*Overview of the OIGA2A framework.*

---

## Results

### Comparison with Existing Methods

Our method employs a **single unified Any-to-Any model** for all 12 tasks, whereas all baseline methods require training **12 independent One-to-One models**.

#### (a) DEM → ANY

| Method | DEM→NIR FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | DEM→RGB FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | DEM→SPOT FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pix2Pix (CVPR, 2017) | 367.1304 | 0.7081 | 13.6182 | 0.2387 | 272.0429 | 0.7273 | **14.3057** | 0.3532 | 176.7447 | 0.1125 | 29.8107 | 0.8483 |
| Pix2PixHD (CVPR, 2018) | 281.2315 | 0.6777 | 13.4883 | 0.3506 | 287.3042 | 0.7109 | 13.6673 | 0.3729 | 119.9822 | **0.0768** | **32.2567** | **0.9157** |
| StegoGAN (CVPR, 2024) | 263.6834 | 0.7296 | 12.8863 | 0.4589 | 271.8155 | 0.7988 | 11.3851 | 0.3649 | 143.9765 | 0.1295 | 29.0394 | 0.8066 |
| FCDiffusion (AAAI, 2024) | 396.8576 | 0.9158 | 9.5075 | 0.2862 | 340.8000 | 0.9860 | 9.5466 | 0.3492 | 256.0318 | 0.5029 | 13.1507 | 0.5649 |
| CDTSDE (ICLR, 2026) | 440.6330 | 0.8625 | 11.5615 | 0.2663 | 391.3203 | 0.7959 | 9.4740 | 0.1247 | 335.5845 | 1.1253 | 4.6325 | 0.0582 |
| LBM (ICCV, 2025) | 355.8055 | 0.9036 | 12.1598 | **0.4865** | 327.1792 | 0.8987 | 13.5023 | **0.5102** | 301.4901 | 0.6673 | 9.2328 | 0.2866 |
| Dual-App-Bridge (CVPR, 2025) | 358.3385 | 0.7900 | 12.0918 | 0.3733 | 321.8264 | 0.7889 | 12.9115 | 0.4189 | 307.8813 | 0.8493 | 7.5977 | 0.1818 |
| **Ours** | **258.8817** | **0.5419** | **14.6291** | 0.4027 | **228.8502** | **0.6600** | 12.7016 | 0.3732 | **98.4160** | 0.0991 | 30.0675 | 0.8771 |

#### (b) NIR → ANY

| Method | NIR→DEM FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | NIR→RGB FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | NIR→SPOT FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pix2Pix (CVPR, 2017) | 217.7814 | 0.7055 | **16.3526** | 0.6199 | 293.9488 | 0.6554 | 13.9647 | 0.3528 | 182.5380 | 0.3095 | 19.0610 | 0.6870 |
| Pix2PixHD (CVPR, 2018) | 147.3903 | 0.3944 | 15.7957 | **0.9270** | 227.4111 | 0.6653 | 14.8828 | 0.4184 | **97.7339** | **0.0768** | **32.1554** | **0.9164** |
| StegoGAN (CVPR, 2024) | 346.3830 | 0.9165 | 12.7144 | 0.5367 | 115.2302 | 0.4161 | 14.0133 | 0.6723 | 117.5049 | 0.1084 | 26.8914 | 0.3696 |
| FCDiffusion (AAAI, 2024) | 220.4016 | 0.4984 | 10.2435 | 0.7326 | 358.5142 | 0.9222 | 9.1056 | 0.3153 | 244.1984 | 0.5488 | 10.9218 | 0.4064 |
| CDTSDE (ICLR, 2026) | 324.5986 | 1.0362 | 11.5687 | 0.2929 | 372.1441 | 0.8467 | 8.0781 | 0.1728 | 368.4434 | 1.1323 | 6.2722 | 0.0639 |
| LBM (ICCV, 2025) | 362.9729 | 0.9045 | 11.4484 | 0.5648 | **104.7898** | **0.3423** | **15.5496** | **0.7128** | 314.1208 | 0.8173 | 0.9629 | 0.0902 |
| Dual-App-Bridge (CVPR, 2025) | 295.8813 | 1.0167 | 12.4218 | 0.4318 | 277.2984 | 0.6890 | 13.3723 | 0.3485 | 366.2873 | 1.0632 | 8.7943 | 0.1451 |
| **Ours** | **96.5765** | **0.3533** | 15.2649 | 0.9324 | 212.2234 | 0.5904 | 14.0881 | 0.3927 | 115.6396 | 0.0863 | 32.0635 | 0.9162 |

#### (c) RGB → ANY

| Method | RGB→DEM FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | RGB→NIR FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | RGB→SPOT FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pix2Pix (CVPR, 2017) | 209.8003 | 0.6046 | **16.4602** | 0.8122 | 403.9548 | 0.6284 | 16.9690 | 0.3357 | 165.5936 | 0.1034 | 32.3151 | 0.9019 |
| Pix2PixHD (CVPR, 2018) | 185.2071 | 0.3701 | 16.4481 | 0.8540 | 247.8943 | 0.5703 | 16.2233 | 0.4299 | **94.2014** | 0.0679 | 32.9161 | 0.9234 |
| StegoGAN (CVPR, 2024) | 337.5476 | 0.9007 | 13.6483 | 0.5252 | 173.1464 | 0.4402 | 14.5769 | 0.7048 | 129.4041 | 0.1821 | 27.7744 | 0.6831 |
| FCDiffusion (AAAI, 2024) | 183.9098 | 0.4061 | 14.9057 | 0.9249 | 324.8305 | 0.8411 | 11.5574 | 0.3199 | 198.7715 | 0.4008 | 12.4607 | 0.5551 |
| CDTSDE (ICLR, 2026) | 302.9182 | 0.5753 | 6.3171 | 0.1681 | 557.3730 | 0.9949 | 8.1969 | 0.1566 | 362.2896 | 1.1299 | 4.3720 | 0.0371 |
| LBM (ICCV, 2025) | 195.8215 | 0.6406 | 14.0168 | 0.8135 | **154.1106** | **0.3413** | **17.4772** | **0.7202** | 219.9140 | 0.0911 | 32.1548 | 0.9204 |
| Dual-App-Bridge (CVPR, 2025) | 296.9056 | 1.0171 | 13.5553 | 0.3696 | 366.0795 | 0.6802 | 13.9759 | 0.3513 | 328.4411 | 1.0999 | 8.3544 | 0.1245 |
| **Ours** | **87.1243** | **0.3433** | 16.3214 | **0.9378** | 230.1838 | 0.4957 | 16.0604 | 0.3892 | 106.9237 | **0.0593** | **33.0879** | **0.9240** |

#### (d) SPOT → ANY

| Method | SPOT→DEM FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | SPOT→NIR FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ | SPOT→RGB FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pix2Pix (CVPR, 2017) | 293.1730 | 0.8780 | **15.9983** | 0.3083 | 326.7645 | 0.6933 | **14.2128** | 0.3220 | 314.3107 | 0.8278 | **14.7264** | 0.3701 |
| Pix2PixHD (CVPR, 2018) | 156.1668 | 0.4075 | 14.7992 | 0.9066 | 291.7448 | 0.6920 | 13.4232 | 0.3710 | 247.7035 | 0.6992 | 13.2611 | 0.3979 |
| StegoGAN (CVPR, 2024) | 267.3759 | 0.6241 | 6.4949 | 0.2068 | 424.3611 | 1.0139 | 7.5584 | 0.0228 | 360.2431 | 1.0138 | 6.7281 | 0.0620 |
| FCDiffusion (AAAI, 2024) | 152.5789 | **0.3885** | 14.7436 | **0.9221** | 352.5077 | 0.8596 | 10.7871 | 0.3020 | 388.1550 | 0.9271 | 11.4776 | **0.4536** |
| CDTSDE (ICLR, 2026) | 328.6274 | 1.0330 | 11.5363 | 0.1732 | 406.3377 | 0.7710 | 11.0487 | 0.2269 | 364.0502 | 0.7723 | 11.1651 | 0.1765 |
| LBM (ICCV, 2025) | 182.8692 | 0.4173 | 13.0116 | 0.8769 | 511.5784 | 0.9427 | 9.8985 | 0.3116 | 425.2980 | 0.9063 | 11.1131 | 0.4044 |
| Dual-App-Bridge (CVPR, 2025) | 274.4586 | 0.9734 | 9.7430 | 0.5091 | 347.7046 | 0.7404 | 10.1213 | 0.3094 | 303.5912 | 0.7438 | 9.7162 | 0.3334 |
| **Ours** | **119.8916** | 0.4057 | 15.0068 | 0.9204 | **238.6052** | **0.5822** | 14.2072 | **0.3741** | **239.1283** | **0.6393** | 13.2619 | 0.4030 |

### Visualization

![Visualization](figures/visualization.jpg)

*Visualization of translation results across all modality pairs.*
