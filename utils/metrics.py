from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import torch.nn.functional as F


def calculate_psnr(pred, target, psnr):
    if psnr is None:
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    psnr.update(pred, target)
    return psnr

def calculate_ssim(pred, target, ssim):
    if ssim is None:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim.update(pred, target)
    return ssim

def calculate_fid(pred, target, fid):
    if fid is None:
        fid = FrechetInceptionDistance(normalize=True).to(pred.device)
    with torch.autocast("cuda", torch.bfloat16 if pred.dtype == torch.bfloat16 else torch.float32):
        if pred.shape[1] == 1:
            fid.update(pred.repeat(1, 3, 1, 1), real=False)
            fid.update(target.repeat(1, 3, 1, 1), real=True)
        else:
            fid.update(pred, real=False)
            fid.update(target, real=True)
    return fid

def calculate_lpips(pred, target, lpips, net_type="alex"):
    if lpips is None:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True).to(pred.device)
    if pred.shape[1] == 1:
        lpips.update(pred.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1))
    else:
        lpips.update(pred, target)
    return lpips


class DirectionalCLIPSimilarity:
    """Directional CLIP Similarity (DCLIP-S) for cross-modal translation evaluation.

    Measures whether the semantic direction from source to generated image
    matches the direction from source to target image in CLIP embedding space.

    d_ref = CLIP(target) - CLIP(source)   # reference direction
    d_gen = CLIP(generated) - CLIP(source) # generated direction
    DCLIP-S = cosine_similarity(d_ref, d_gen)

    Unlike standard CLIP similarity (which measures absolute alignment),
    DCLIP-S measures RELATIVE semantic correctness — whether the model
    moves in the right direction in semantic space.
    """

    def __init__(self, clip_model_name="ViT-L/14", device="cuda"):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Accumulate directions for averaging
        self.directions_ref = []
        self.directions_gen = []

    def _encode_image(self, img_tensor):
        """Encode image tensor (B, C, H, W) in [0,1] to CLIP image embeddings."""
        # Resize to 224x224 for CLIP
        img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # CLIP expects [0,1] float, normalize with CLIP's mean/std
        img_clip = self.preprocess.transforms[-1](img_resized)  # Apply CLIP normalize
        with torch.no_grad():
            features = self.model.encode_image(img_clip)
        return F.normalize(features.float(), dim=-1)

    def update(self, source, generated, target):
        """Update with a batch of source, generated, and target images.

        Args:
            source: (B, C, H, W) source image tensor in [0,1]
            generated: (B, C, H, W) generated image tensor in [0,1]
            target: (B, C, H, W) target image tensor in [0,1]
        """
        e_src = self._encode_image(source)
        e_gen = self._encode_image(generated)
        e_tgt = self._encode_image(target)

        # Compute directions
        d_ref = e_tgt - e_src  # reference direction
        d_gen = e_gen - e_src  # generated direction

        self.directions_ref.append(d_ref.cpu())
        self.directions_gen.append(d_gen.cpu())

    def compute(self):
        """Compute the mean directional CLIP similarity across all accumulated samples.

        Returns:
            float: Mean cosine similarity between reference and generated directions.
                   1.0 = perfect directional alignment, 0.0 = orthogonal, -1.0 = opposite.
        """
        if not self.directions_ref:
            return 0.0

        all_ref = torch.cat(self.directions_ref, dim=0)
        all_gen = torch.cat(self.directions_gen, dim=0)

        # Cosine similarity per sample, then average
        cos_sim = F.cosine_similarity(all_ref, all_gen, dim=-1)
        return cos_sim.mean().item()

    def reset(self):
        """Reset accumulated directions."""
        self.directions_ref = []
        self.directions_gen = []