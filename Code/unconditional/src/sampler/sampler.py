import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)


_SAMPLERS = {
    "ddim":         DDIMScheduler,
    "ddpm":         DDPMScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
    "unipc":        UniPCMultistepScheduler,
}


def make_scheduler(sampler, beta_schedule, prediction_type, num_train_timesteps=250):
    cls = _SAMPLERS[sampler]

    kwargs = {
        "num_train_timesteps": num_train_timesteps,
        "prediction_type": "v_prediction" if prediction_type == "v" else "epsilon",
    }
    if beta_schedule == "cosine":
        kwargs["beta_schedule"] = "squaredcos_cap_v2"
    else:
        kwargs["beta_schedule"] = "linear"
        kwargs["beta_start"] = 1e-4
        kwargs["beta_end"] = 0.02

    if sampler in ("ddim", "ddpm"):
        kwargs["clip_sample"] = False

    return cls(**kwargs)


@torch.no_grad()
def sample(model, scheduler, image_size,
           num_samples=1, num_steps=50, device="cuda", seed=None,
           cond=None, size_cond=None, guidance_scale=1.0):
    # set_timesteps must be called every time — multistep solvers (DPM++/UniPC)
    # carry state across calls and fail on the second sample otherwise.
    scheduler.set_timesteps(num_steps)
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    x = torch.randn(num_samples, 1, image_size, image_size, image_size, device=device)

    # collect whichever conditioning(s) the caller provided
    cond_kwargs = {}
    if cond is not None:
        cond_kwargs["cond"] = cond
    if size_cond is not None:
        cond_kwargs["size_cond"] = size_cond
    use_cfg = guidance_scale > 1.0 and cond_kwargs

    for t in scheduler.timesteps:
        t_batch = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if use_cfg:
                # CFG: blend conditional with null-conditional prediction
                null_kwargs = {k: None for k in cond_kwargs}
                pred_cond = model(x, t_batch, **cond_kwargs).float()
                pred_uncond = model(x, t_batch, **null_kwargs).float()
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = model(x, t_batch, **cond_kwargs).float()
        x = scheduler.step(pred, t, x).prev_sample

    return x.clamp(-1, 1)
