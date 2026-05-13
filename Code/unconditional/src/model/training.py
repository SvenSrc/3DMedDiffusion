import math
import subprocess
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---- small helpers used inside the driver loop ----

def get_gpu_memory_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        return None
    return None


def compute_weight_norm(model):
    sq = sum(p.data.norm(2).item() ** 2 for p in model.parameters())
    return sq ** 0.5


def snapshot_weights(model):
    return {name: p.data.clone() for name, p in model.named_parameters()}


def compute_weight_update_ratio(model, snapshot):
    upd_sq = sum((p.data - snapshot[name]).norm(2).item() ** 2
                 for name, p in model.named_parameters() if name in snapshot)
    wt_sq = sum(p.data.norm(2).item() ** 2 for p in model.parameters())
    return (upd_sq / wt_sq) ** 0.5 if wt_sq > 0 else 0.0


def compute_ema_divergence(model, ema):
    sq = sum((p.data - ema.shadow[name]).norm(2).item() ** 2
             for name, p in model.named_parameters() if name in ema.shadow)
    return sq ** 0.5


def get_lr_with_warmup(step, warmup_steps, base_lr, total_steps):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ---- the training step and validation pass ----

def train_one_epoch(model, loader, optimizer, noise_scheduler, scaler, ema,
                    timestep_sampler, config, global_step, total_steps):
    model.train()
    total_loss = 0.0
    data_time = fwd_time = bwd_time = 0.0
    grad_norms = []
    grad_clip_count = 0
    nan_inf_count = 0
    loss_buckets = {0: [], 1: [], 2: [], 3: [], 4: []}
    batch_end = time.time()

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        data_time += time.time() - batch_end
        batch = batch.to(config.device)

        lr = get_lr_with_warmup(global_step, config.warmup_steps, config.learning_rate, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if config.use_loss_aware_sampling and global_step > config.warmup_steps:
            t, imp_weights = timestep_sampler.sample(batch.shape[0], config.device)
        else:
            t = torch.randint(0, config.num_timesteps, (batch.shape[0],), device=config.device)
            imp_weights = torch.ones(batch.shape[0], device=config.device)

        noise = torch.randn_like(batch)
        x_t = noise_scheduler.add_noise(batch, t, noise)
        optimizer.zero_grad()

        fwd_start = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(x_t, t)
            target = noise_scheduler.get_velocity(batch, noise, t) if config.prediction_type == "v" else noise
            loss_per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3, 4))
            if config.snr_gamma > 0:
                snr_w = noise_scheduler.get_snr_weights(t, config.snr_gamma).squeeze()
                loss_per_sample = loss_per_sample * snr_w
            loss = (loss_per_sample * imp_weights).mean()
        fwd_time += time.time() - fwd_start

        # per-timestep raw losses for stats bucketing and the loss-aware sampler
        with torch.no_grad():
            raw = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3, 4))
            for ti, li in zip(t.cpu().tolist(), raw.cpu().tolist()):
                loss_buckets[min(ti // 50, 4)].append(li)
        if config.use_loss_aware_sampling and torch.isfinite(raw).all():
            timestep_sampler.update_losses(t, raw)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_inf_count += 1

        bwd_start = time.time()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        grad_norms.append(grad_norm.item())
        if grad_norm.item() >= config.grad_clip:
            grad_clip_count += 1
        scaler.step(optimizer)
        scaler.update()
        bwd_time += time.time() - bwd_start

        ema.update(model)
        total_loss += loss.item()
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        batch_end = time.time()

    return {
        "loss":             total_loss / len(loader),
        "global_step":      global_step,
        "data_time":        data_time,
        "fwd_time":         fwd_time,
        "bwd_time":         bwd_time,
        "grad_norm_avg":    sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
        "grad_norm_max":    max(grad_norms) if grad_norms else 0.0,
        "grad_clip_count":  grad_clip_count,
        "nan_inf_count":    nan_inf_count,
        "loss_buckets":     {k: (sum(v) / len(v) if v else 0.0) for k, v in loss_buckets.items()},
    }


@torch.no_grad()
def validate(model, loader, noise_scheduler, config):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(config.device)
        t = torch.randint(0, config.num_timesteps, (batch.shape[0],), device=config.device)
        noise = torch.randn_like(batch)
        x_t = noise_scheduler.add_noise(batch, t, noise)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(x_t, t)
            target = noise_scheduler.get_velocity(batch, noise, t) if config.prediction_type == "v" else noise
            loss = F.mse_loss(pred, target)
        total_loss += loss.item()
    return total_loss / len(loader)
