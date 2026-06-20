# 3DMedDiffusion

3D diffusion model for medical image synthesis, focusing on brain MRI generation from the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/621282) dataset. Implements unconditional and conditional variants with a full tumor generation pipeline.

This project was developed as part of a research paper (T3200) at DHBW Mosbach/Bad-Mergentheim.

This repo is free to use.

---

## Overview

The core idea is a 3D UNet-based denoising diffusion probabilistic model (DDPM) that operates on volumetric brain MRI data. Two conditional extensions are built on top:

- **seg2mri**: generates a realistic brain MRI conditioned on a tumor segmentation mask
- **class2mask**: generates a plausible tumor segmentation mask conditioned on a size class (small / medium / large)

Combined into a **pipeline**: given a desired tumor size class, the model first generates a segmentation map, places it anatomically and then generates a full brain MRI around it.

---

## Architecture

**3D UNet** with ~156M parameters:

| Component | Detail |
|---|---|
| Spatial dims | 32³ (primary) and 48³ |
| Channel multipliers | (1, 2, 4, 4) |
| Residual blocks | 2 per resolution |
| Self-attention | 4-head, at resolutions 8 and 4 |
| Time embedding | Sinusoidal → 2-layer MLP |
| Normalization | GroupNorm (8 groups) |
| Activation | SiLU |
| Conditioning | Channel-concat (spatial) + embedding (class) |

**Diffusion setup:**

| Setting | Value |
|---|---|
| Timesteps | 250 |
| Noise schedule | Cosine (Nichol & Dhariwal 2021) |
| Prediction type | v-prediction |
| Loss weighting | Min-SNR (γ = 5.0) |
| EMA decay | 0.9999 |
| Timestep sampling | Loss-aware biased sampling |

**Samplers supported:** DDPM - DDIM - DPM-Solver++ - UniPC

---

## Dataset

[BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/621282) — 1,251 brain tumor MRI scans (T1n modality).

**Preprocessing:** percentile clipping (0.5–99.5%) → crop foreground → pad to cube → resize to 32³ → normalize to [−1, 1].

**Split:** 1,126 train / 125 validation (90/10).

---

## Project Structure

```
3DMedDiffusion/
├── Code/
│   ├── unconditional/          # Baseline models (32³ and 48³)
│   │   ├── 32/                 # Training notebooks
│   │   ├── 48/
│   │   ├── generate.ipynb      # Sample from a trained model
│   │   └── src/
│   │       ├── data/           # Dataset loaders & augmentation
│   │       ├── diffusion/      # Noise schedule
│   │       ├── model/          # UNet3D, residual blocks, EMA, training loop
│   │       ├── sampler/        # DDPM/DDIM/DPM-Solver++/UniPC wrappers
│   │       └── utility/        # Stats & checkpoint tracking
│   ├── conditional/
│   │   ├── train_seg2mri.ipynb         # Train segmentation→MRI model
│   │   ├── train_class2mask.ipynb      # Train size-class→segmentation model
│   │   ├── generate_conditional.ipynb  # Sample from conditional models
│   │   ├── generate_pipeline.ipynb     # End-to-end tumor generation pipeline
│   │   └── src/
│   │       └── conditional_datasets.py
│   ├── data_processing/        # NIfTI preprocessing scripts
│   ├── metrics/                # SSIM, PSNR, LPIPS, FID, Dice
│   └── figures/                # Visualization & figure notebooks
├── Values/
│   ├── Model Training Values/      # Training configs, loss curves, stats
│   ├── Sample Generation Values/   # Generated .nii.gz samples per sampler
│   ├── Sample Evaluation Results/  # Computed evaluation metrics
│   └── Tumor Sizes/                # TNM-based size classification labels
└── T3200/                      # Thesis LaTeX source
```

---

## Training

Each model variant is trained via its corresponding notebook in `Code/unconditional/32/` or `Code/conditional/`.

**Key hyperparameters (baseline):**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Batch size | 10 |
| Epochs | 300 |
| Warmup steps | 500 |
| Gradient clipping | 1.0 |
| CFG dropout (conditional) | 0.15 |

Trained on an NVIDIA GTX 1080 Ti (11.7 GB). The baseline 32³ model takes approximately **56 hours**.

Checkpoints and training stats (loss curves, gradient norms, timestep-bucketed losses) are saved under `Values/Model Training Values/`.

---

## Experiments

Ablation study comparing the following variants at 32³ resolution:

| Variant | Change from baseline |
|---|---|
| **Baseline** | v-prediction · cosine schedule · min-SNR · loss-aware sampling · attention |
| Data Augmentation | + random left-right flip (p = 0.5) |
| Linear Schedule | Cosine → linear beta schedule |
| Noise Prediction | v-prediction → epsilon prediction |
| No Attention | Self-attention blocks removed |
| **48³ Baseline** | Higher resolution |

All four samplers (DDPM, DDIM, DPM-Solver++, UniPC) are evaluated on each variant.

---

## Evaluation

Metrics computed in `Code/metrics/`:

- **SSIM** and **PSNR** — voxel-level structural fidelity
- **LPIPS** — perceptual similarity (AlexNet, per-slice)
- **FID** — distribution distance (Inception V3, per axial slice)
- **Dice Score** — segmentation quality (per tumor class, NaN for absent classes)

All metrics use `data_range=2.0` to account for [−1, 1] normalization.

---

## Pipeline

The end-to-end tumor generation pipeline (`generate_pipeline.ipynb`):

1. **class2mask** generates a tumor segmentation conditioned on a size class (T1 ≤2cm / T2 2–5cm / T3 ≥5cm)
2. The segmentation is placed at the center-of-mass of a reference BraTS segmentation
3. **seg2mri** generates a full brain MRI conditioned on the placed segmentation

Classifier-Free Guidance with `guidance_scale=3.0` is applied at each conditional stage.

---

## Requirements

The notebooks use standard PyTorch and medical imaging libraries. A `environment.json` with the exact package versions used for each experiment is saved alongside each model's training config.

Key dependencies: `torch` · `nibabel` · `diffusers` · `torchmetrics` · `lpips`

---

## To-do

[] Translate into english

---

## Version

1.0 - Finalizing, no major changes will occur to the code and paper itself.