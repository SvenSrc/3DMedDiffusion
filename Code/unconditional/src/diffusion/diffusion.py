import math

import torch
import torch.nn.functional as F


# Cosine schedule from Nichol & Dhariwal 2021 
# Linear schedule from Ho et al. 2020


def cosine_betas(num_timesteps, s=0.008):
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_betas(num_timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)
