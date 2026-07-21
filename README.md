<div align="center">

# OIGA2A: Open Instruction Guided Any-to-Any Aerial Image Translation via a Single Flow Matching Model

**Anonymous Authors** | Under Review at AAAI 2027

> *The code will be released upon paper acceptance to foster reproducibility in the remote sensing community.*

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-red)](https://anonymous.4open.science/r/OIGA2A-XXXX)
[![Framework](https://img.shields.io/badge/Framework-Flow%20Matching-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

---

## Abstract
Existing aerial image cross-modal translation methods typically train separate models for each source-target modality pair, requiring full retraining for every new modality combination. This results in quadratically increasing total model size and deployment complexity as the number of modalities grows, severely limiting their scalability in practical multi-modal remote sensing applications.

To address this limitation, we propose **Open Instruction Guided Any-to-Any (OIGA2A)**, a single weight framework enabling open instruction guided **any-to-any** translation among RGB, Near-Infrared (NIR), Digital Elevation Models (DEM) and SPOT satellite imagery. It leverages a **flow matching** diffusion Transformer operating in a shared latent space learned by a single jointly trained KL-regularized Variational Autoencoder (VAE). A dynamic loss mask router adaptively applies L<sub>1</sub>+perceptual loss for optical texture modalities and spatial gradient consistency loss for topographic DEM, ensuring faithful reconstruction while preserving unique modality characteristics.

A conditioning scheme integrates open instruction semantics via a LoRA finetuned CLIP encoder and structural alignment via AdaLN-Zero modulation, with classifier free guidance further enhancing inference controllability. Extensive experiments across all 12 possible translation directions demonstrate that our **single unified model** outperforms pair-specific baselines on average across most quantitative metrics, while consolidating all directions into a single deployable weight set and qualitatively demonstrating faithful open-vocabulary instruction following. By consolidating all mappings into one compact model, OIGA2A significantly reduces deployment complexity and storage footprint, enabling efficient deployment for real world remote sensing applications.

---

## Contributions

<div align="center">

![Qualitative comparison](figures/introduction.jpg)

*Qualitative comparison between traditional image translation methods and our method.*

</div>

The primary contributions are:

- We propose **OIGA2A**, the first open instruction guided any-to-any translation framework for remote sensing, condensing all 12 bidirectional cross-platform pathways into a single unified weight set and eliminating the quadratic model growth of pairwise methods.

- We design a unified architecture that tightly couples three components into a single coherent design:
  - **Dynamic Loss Mask Router** — within a single shared KL-VAE, which adaptively routes optical textural losses and geometric gradient consistency constraints per modality, eliminating cross-modal gradient interference while simultaneously preserving photometric fidelity and topographic structure.
  - **Modality Calibrated Flow Matching** — which augments a shared Scalable Interpolant Transformer (SiT) backbone with per-modality latent normalization and lightweight per-target-modality residual adapters, allowing a single flow matching backbone to accommodate the heterogeneous latent statistics of all modalities without sacrificing weight sharing or inference efficiency.
  - **Conditioning Strategy** — that integrates a LoRA-adapted CLIP instruction encoder with AdaLN-Zero modulation jointly conditioned on source modality, target modality, and timestep, enabling simultaneous satisfaction of open expert-level semantic intent and physical modality constraints.

- Extensive evaluations on all 12 translation permutations of the FLAIR-HUB benchmark show that a single OIGA2A model consistently outperforms seven pairwise trained baselines, reducing average FID by **14.7%** relative to the strongest baseline (Pix2PixHD) while achieving the best average LPIPS and SSIM, using **3&times;&ndash;40&times;** fewer total parameters than five of the seven pairwise baseline sets (Pix2PixHD, FCDiffusion, CDTSDE, LBM, Dual-App-Bridge); against the two lightweight GAN baselines, OIGA2A&rsquo;s single model is comparable in parameter count, though it still collapses 6&ndash;12 separately managed checkpoints into one.

---

## Method

<div align="center">

![Overview of the OIGA2A framework](figures/overview.jpg)

*Overview of the OIGA2A framework.*

</div>

---

## 📊 Results

### Comparison with Existing Methods
*Note: Our method employs a **single unified Any-to-Any model** for all 12 tasks, whereas all baseline methods require training **12 independent One-to-One models**.*

#### (a) Source Modality: DEM → ANY
<!-- 使用 HTML 表格并加宽 Method 列 -->

<table align="center" style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background-color: #2c3e50; color: white;">
      <th rowspan="2" style="text-align: left; padding: 10px; min-width: 190px; border: 1px solid #2c3e50;">Method</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">DEM &rarr; NIR</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">DEM &rarr; RGB</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">DEM &rarr; SPOT</th>
    </tr>
    <tr style="background-color: #2c3e50; color: white;">
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2Pix (CVPR '17)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">367.1304</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7081</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.6182</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2387</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">272.0429</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7273</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>14.3057</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3532</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">176.7447</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1125</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">29.8107</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8483</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2PixHD (CVPR '18)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">281.2315</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.6777</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.4883</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3506</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">287.3042</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7109</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.6673</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3729</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>119.9822</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.0768</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>32.2567</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.9157</b></td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">StegoGAN (CVPR '24)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>263.6834</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7296</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.8863</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.4589</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>271.8155</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7988</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.3851</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3649</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">143.9765</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1295</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">29.0394</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8066</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">FCDiffusion (AAAI '24)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">396.8576</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9158</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.5075</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2862</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">340.8000</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9860</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.5466</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3492</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">256.0318</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5029</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.1507</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5649</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">CDTSDE (ICLR '26)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">440.6330</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8625</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.5615</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2663</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">391.3203</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7959</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.4740</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1247</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">335.5845</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.1253</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">4.6325</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0582</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">LBM (ICCV '25)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">355.8055</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9036</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.1598</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.4865</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">327.1792</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8987</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.5023</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.5102</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">301.4901</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6673</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.2328</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2866</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Dual-App-Bridge (CVPR '25)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">358.3385</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7900</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.0918</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3733</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">321.8264</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7889</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.9115</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.4189</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">307.8813</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8493</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">7.5977</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1818</td>
    </tr>
    <tr style="background-color: #e8f4fd; font-weight: 600;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>Ours</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>258.8817</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.5419</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>14.6291</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4027</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>228.8502</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.6600</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.7016</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3732</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>98.4160</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.0991</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>30.0675</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.8771</u></td>
    </tr>
  </tbody>
</table>

<br>

#### (b) Source Modality: NIR → ANY
<table align="center" style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background-color: #2c3e50; color: white;">
      <th rowspan="2" style="text-align: left; padding: 10px; min-width: 190px; border: 1px solid #2c3e50;">Method</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">NIR &rarr; DEM</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">NIR &rarr; RGB</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">NIR &rarr; SPOT</th>
    </tr>
    <tr style="background-color: #2c3e50; color: white;">
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2Pix (CVPR, 2017)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">217.7814</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7055</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>16.3526</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6199</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">293.9488</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6554</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.9647</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3528</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">182.5380</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3095</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">19.0610</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6870</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2PixHD (CVPR, 2018)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>147.3903</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.3944</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>15.7957</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.9270</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">227.4111</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6653</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.8828</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4184</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>97.7339</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.0768</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>32.1554</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.9164</b></td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">StegoGAN (CVPR, 2024)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">346.3830</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9165</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.7144</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5367</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>115.2302</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.4161</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.0133</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.6723</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>117.5049</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.1084</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>26.8914</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.3696</u></td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">FCDiffusion (AAAI, 2024)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">220.4016</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4984</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">10.2435</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7326</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">358.5142</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9222</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.1056</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3153</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">244.1984</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5488</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">10.9218</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4064</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">CDTSDE (ICLR, 2026)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">324.5986</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0362</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.5687</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2929</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">372.1441</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8467</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">8.0781</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1728</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">368.4434</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.1323</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">6.2722</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0639</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">LBM (ICCV, 2025)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">362.9729</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9045</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.4484</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5648</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>104.7898</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.3423</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>15.5496</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.7128</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">314.1208</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8173</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9629</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0902</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Dual-App-Bridge (CVPR, 2025)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">295.8813</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0167</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.4218</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4318</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">277.2984</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6890</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.3723</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3485</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">366.2873</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0632</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">8.7943</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1451</td>
    </tr>
    <tr style="background-color: #e8f4fd; font-weight: 600;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>Ours</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>96.5765</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.3533</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">15.2649</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.9324</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">212.2234</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5904</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>14.0881</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3927</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">115.6396</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0863</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">32.0635</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9162</td>
    </tr>
  </tbody>
</table>

<br>

#### (c) Source Modality: RGB → ANY
<table align="center" style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background-color: #2c3e50; color: white;">
      <th rowspan="2" style="text-align: left; padding: 10px; min-width: 190px; border: 1px solid #2c3e50;">Method</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">RGB &rarr; DEM</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">RGB &rarr; NIR</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">RGB &rarr; SPOT</th>
    </tr>
    <tr style="background-color: #2c3e50; color: white;">
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2Pix (CVPR, 2017)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">209.8003</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6046</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>16.4602</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8122</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">403.9548</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6284</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>16.9690</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3357</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">165.5936</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1034</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">32.3151</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9019</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2PixHD (CVPR, 2018)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">185.2071</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.3701</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>16.4481</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8540</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">247.8943</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5703</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">16.2233</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4299</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>94.2014</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.0679</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>32.9161</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.9234</u></td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">StegoGAN (CVPR, 2024)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">337.5476</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9007</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.6483</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5252</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>173.1464</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.4402</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.5769</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.7048</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">129.4041</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1821</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">27.7744</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6831</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">FCDiffusion (AAAI, 2024)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>183.9098</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4061</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.9057</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.9249</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">324.8305</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8411</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.5574</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3199</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">198.7715</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4008</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">12.4607</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5551</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">CDTSDE (ICLR, 2026)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">302.9182</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5753</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">6.3171</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1681</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">557.3730</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9949</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">8.1969</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1566</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">362.2896</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.1299</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">4.3720</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0371</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">LBM (ICCV, 2025)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">195.8215</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6406</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.0168</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8135</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>154.1106</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.3413</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>17.4772</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.7202</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">219.9140</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0911</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">32.1548</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9204</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Dual-App-Bridge (CVPR, 2025)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">296.9056</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0171</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.5553</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3696</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">366.0795</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6802</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.9759</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3513</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">328.4411</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0999</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">8.3544</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1245</td>
    </tr>
    <tr style="background-color: #e8f4fd; font-weight: 600;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>Ours</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>87.1243</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.3433</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">16.3214</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.9378</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">230.1838</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4957</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">16.0604</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3892</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>106.9237</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.0593</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>33.0879</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.9240</b></td>
    </tr>
  </tbody>
</table>

<br>

#### (d) Source Modality: SPOT → ANY
<table align="center" style="width: 100%; border-collapse: collapse; font-size: 14px;">
  <thead>
    <tr style="background-color: #2c3e50; color: white;">
      <th rowspan="2" style="text-align: left; padding: 10px; min-width: 190px; border: 1px solid #2c3e50;">Method</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">SPOT &rarr; DEM</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">SPOT &rarr; NIR</th>
      <th colspan="4" style="text-align: center; padding: 10px; border: 1px solid #2c3e50;">SPOT &rarr; RGB</th>
    </tr>
    <tr style="background-color: #2c3e50; color: white;">
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
      <th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">FID&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">LPIPS&darr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">PSNR&uarr;</th><th style="text-align: center; padding: 6px; border: 1px solid #2c3e50;">SSIM&uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2Pix (CVPR, 2017)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">293.1730</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8780</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>15.9983</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3083</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">326.7645</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6933</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>14.2128</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3220</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">314.3107</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8278</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>14.7264</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3701</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Pix2PixHD (CVPR, 2018)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">156.1668</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4075</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.7992</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9066</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>291.7448</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.6920</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.4232</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.3710</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>247.7035</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.6992</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.2611</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3979</td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">StegoGAN (CVPR, 2024)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">267.3759</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.6241</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">6.4949</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2068</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">424.3611</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0139</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">7.5584</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0228</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">360.2431</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0138</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">6.7281</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.0620</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">FCDiffusion (AAAI, 2024)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>152.5789</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.3885</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">14.7436</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.9221</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">352.5077</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8596</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">10.7871</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3020</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">388.1550</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9271</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.4776</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.4536</b></td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">CDTSDE (ICLR, 2026)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">328.6274</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">1.0330</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.5363</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1732</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">406.3377</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7710</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.0487</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.2269</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">364.0502</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7723</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.1651</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.1765</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">LBM (ICCV, 2025)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">182.8692</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4173</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">13.0116</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.8769</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">511.5784</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9427</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.8985</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3116</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">425.2980</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9063</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">11.1131</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.4044</u></td>
    </tr>
    <tr>
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;">Dual-App-Bridge (CVPR, 2025)</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">274.4586</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.9734</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.7430</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.5091</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">347.7046</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7404</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">10.1213</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3094</td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">303.5912</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.7438</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">9.7162</td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.3334</td>
    </tr>
    <tr style="background-color: #e8f4fd; font-weight: 600;">
      <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>Ours</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>119.8916</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.4057</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>15.0068</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>0.9204</u></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>238.6052</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.5822</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>14.2072</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.3741</b></td>
      <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>239.1283</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><b>0.6393</b></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;"><u>13.2619</u></td><td style="text-align: center; padding: 8px; border-bottom: 1px solid #ecf0f1;">0.4030</td>
    </tr>
  </tbody>
</table>

---

### 🎨 Visualization

<div align="center">

![Visualization](figures/visualization.jpg)

*Visualization of translation results across some modality pairs.*



---

## Citation
If you find our work useful in your research, please consider citing our paper
