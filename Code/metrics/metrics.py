import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Volumes from the diffusion pipeline live in [-1, 1] (see preprocessing in
# data_processing/). data_range=2.0 is the span of that interval and is what
# both skimage SSIM/PSNR need to interpret the values correctly.
DATA_RANGE = 2.0


# --- full-reference, per-pair metrics ------------------------------------

def ssim(real, gen):
    # skimage handles 3D directly when an explicit data_range is given.
    return float(structural_similarity(real, gen, data_range=DATA_RANGE))


def psnr(real, gen):
    return float(peak_signal_noise_ratio(real, gen, data_range=DATA_RANGE))


def load_lpips(device):
    import lpips
    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


@torch.no_grad()
def lpips_score(real, gen, model, device):
    # LPIPS uses a pretrained 2D AlexNet — apply per axial slice and
    # average. Inputs must be [-1, 1] with 3 channels, so replicate the
    # gray slice across RGB.
    a = torch.from_numpy(real).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
    b = torch.from_numpy(gen ).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
    return float(model(a, b).mean().item())


# --- FID -----------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def load_inception(device):
    # Standard Inception V3 ImageNet weights, classifier head replaced with
    # Identity so model() returns the 2048-d pool features used for FID.
    from torchvision.models import Inception_V3_Weights, inception_v3
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    m.fc = torch.nn.Identity()
    return m.to(device).eval()


@torch.no_grad()
def _inception_features(volumes, model, device):
    # Slice each 3D volume axially, feed the slices through Inception, and
    # collect features. Slice-wise FID is the standard approach for 3D
    # medical imaging since Inception itself is 2D.
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    feats = []
    for v in volumes:
        x = torch.from_numpy(v).to(device)             # (D, H, W) in [-1, 1]
        x = (x + 1.0) / 2.0                            # -> [0, 1]
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)          # (D, 3, H, W)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std                           # ImageNet normalization
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def fid(real_volumes, gen_volumes, model, device):
    f_real = _inception_features(real_volumes, model, device)
    f_gen  = _inception_features(gen_volumes,  model, device)

    mu_r, mu_g = f_real.mean(axis=0), f_gen.mean(axis=0)
    cov_r = np.cov(f_real, rowvar=False)
    cov_g = np.cov(f_gen,  rowvar=False)

    # sqrtm of a covariance product can pick up tiny imaginary components
    # from numerical noise — strip them before taking the trace.
    covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


# --- Dice (segmentation overlap) -----------------------------------------

def _discretize(vol, num_classes):
    # Inverse of the {0..K-1} -> [-1, 1] mapping used during preprocessing.
    # Generated seg-maps are continuous, so round to the nearest label.
    x = (vol + 1.0) / 2.0 * (num_classes - 1)
    return np.clip(np.round(x), 0, num_classes - 1).astype(np.int32)


def dice_per_class(real, gen, num_classes=4):
    # Per-class Dice for labels {1..K-1}. Label 0 is background and
    # conventionally excluded in BraTS-style reporting.
    real_d = _discretize(real, num_classes)
    gen_d  = _discretize(gen,  num_classes)
    scores = []
    for k in range(1, num_classes):
        a = real_d == k
        b = gen_d  == k
        denom = a.sum() + b.sum()
        # if the class is absent in both volumes the score is undefined —
        # NaN so it doesn't pull the per-sample mean toward zero.
        scores.append(float("nan") if denom == 0 else 2.0 * np.logical_and(a, b).sum() / denom)
    return scores


def dice(real, gen, num_classes=4):
    return float(np.nanmean(dice_per_class(real, gen, num_classes)))
