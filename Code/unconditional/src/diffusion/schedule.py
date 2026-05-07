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

class NoiseScheduler:

    def __init__(self, num_timesteps, beta_schedule="cosine", device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device

        if beta_schedule == "cosine":
            betas = cosine_betas(num_timesteps)
        else:
            betas = linear_betas(num_timesteps)

        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt().to(device)
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().to(device)
        self.snr = (alphas_cumprod / (1.0 - alphas_cumprod)).to(device)

    def add_noise(self, x_0, t, noise):
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sa * x_0 + so * noise

    def get_velocity(self, x_0, noise, t):
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sa * noise - so * x_0

    def get_snr_weights(self, t, gamma=5.0):
        snr_t = self.snr[t]
        weights = torch.clamp(snr_t, max=gamma) / snr_t
        return weights.view(-1, 1, 1, 1, 1)
