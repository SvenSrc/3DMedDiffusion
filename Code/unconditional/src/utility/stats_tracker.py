import json
import platform
import sys
from datetime import datetime

import torch


class StatsTracker:

    def __init__(self, experiment_meta, config):
        self.stats = {
            "experiment_info": experiment_meta,
            "model_info": {
                "num_parameters": None,
                "num_parameters_millions": None,
                "architecture": "3D UNet",
                "config": {},
            },
            "environment": {
                "gpu_name": "", "gpu_vram_gb": None,
                "cpu_model": "", "ram_total_gb": None,
                "pytorch_version": "", "cuda_version": "",
                "python_version": "", "os_info": "", "random_seed": 42,
            },
            "dataset_info": {
                "dataset_name": "BraTS 2023",
                "total_volumes": None, "train_volumes": None, "val_volumes": None,
                "volume_shape": [32, 32, 32],
                "value_range": [-1.0, 1.0],
                "preprocessing": "Percentile clip (0.5-99.5%), crop foreground, pad to cubic, resize to 32^3, normalize to [-1,1]",
            },
            "training_info": {
                "start_time": "", "end_time": "",
                "total_time_hours": None, "total_time_formatted": "",
                "total_gpu_hours": None,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "device": config.device,
                "peak_memory_gb": None,
            },
            "per_epoch_metrics": {
                "train_losses": [], "val_losses": [],
                "learning_rates": [], "epoch_times_seconds": [], "timestamps": [],
                "gpu_memory_allocated_gb": [], "gpu_memory_reserved_gb": [], "gpu_memory_nvidia_smi_gb": [],
                "data_loading_time_sec": [], "forward_pass_time_sec": [], "backward_pass_time_sec": [],
                "gradient_norms": [], "gradient_norms_max": [],
                "gradient_clip_counts": [], "nan_inf_counts": [],
                "loss_spike_flags": [], "ema_divergences": [],
                "loss_bucket_0_50": [], "loss_bucket_50_100": [],
                "loss_bucket_100_150": [], "loss_bucket_150_200": [], "loss_bucket_200_250": [],
                "weight_norms": [], "weight_update_ratios": [],
            },
            "timing_summary": {
                "average_epoch_time_sec": None, "fastest_epoch_time_sec": None,
                "slowest_epoch_time_sec": None, "median_epoch_time_sec": None,
                "total_data_loading_time_sec": None,
                "total_forward_pass_time_sec": None,
                "total_backward_pass_time_sec": None,
            },
            "convergence": {
                "best_epoch": None, "best_val_loss": None,
                "final_train_loss": None, "final_val_loss": None,
                "convergence_epoch_95pct": None, "overfitting_epoch": None,
            },
            "checkpoints": {
                "periodic_checkpoint_paths": [],
                "best_checkpoint_path": "", "best_checkpoint_epoch": None,
                "final_checkpoint_path": "",
            },
            "checkpoint_strategy": {
                "periodic_at_epochs": [1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
                "save_best_model": True, "save_final_model": True,
                "early_stopping": False, "total_epochs_mandatory": 300,
            },
            "sample_epochs": [1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
            "evaluation_metrics": {
                "image_quality": {
                    "ssim": None, "ssim_std": None, "psnr": None, "psnr_std": None,
                    "mse": None, "mse_std": None, "fid": None, "lpips": None,
                    "num_samples_evaluated": None,
                },
                "diversity": {
                    "sample_diversity_score": None,
                    "pairwise_ssim_mean": None, "pairwise_ssim_std": None,
                    "intensity_histogram_kl_divergence": None,
                    "intensity_histogram_wasserstein": None,
                    "generated_mean": None, "generated_std": None,
                    "generated_min": None, "generated_max": None,
                    "real_mean": None, "real_std": None,
                    "real_min": None, "real_max": None,
                },
                "anatomical_plausibility": {
                    "visual_quality_score_1to5": None,
                    "anatomical_plausibility_score_1to5": None,
                    "slice_consistency_axial": None, "slice_consistency_coronal": None,
                    "slice_consistency_sagittal": None, "slice_consistency_mean": None,
                    "valid_voxel_percentage": None, "notes": "",
                },
                "sampling_performance": {
                    "sampling_time_25_steps_sec": None,
                    "sampling_time_50_steps_sec": None,
                    "sampling_time_100_steps_sec": None,
                    "sampling_time_200_steps_sec": None,
                    "optimal_ddim_steps": None,
                    "ema_vs_noema_ssim_diff": None,
                },
                "denoising_trajectory": {
                    "saved": False,
                    "steps_saved": [250, 200, 150, 100, 50, 25, 10, 0],
                    "seed_used": 42, "path": "",
                },
            },
        }

    def record_environment(self):
        env = self.stats["environment"]
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        env["cpu_model"] = platform.processor()
        env["pytorch_version"] = torch.__version__
        env["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else "N/A"
        env["python_version"] = sys.version.split()[0]
        env["os_info"] = platform.platform()

    def record_dataset(self, dataset, train_size, val_size):
        ds = self.stats["dataset_info"]
        ds["total_volumes"] = len(dataset)
        ds["train_volumes"] = train_size
        ds["val_volumes"] = val_size
        ds["volume_shape"] = list(dataset.volume_shape)

    def record_model(self, model, config):
        n_params = sum(p.numel() for p in model.parameters())
        mi = self.stats["model_info"]
        mi["num_parameters"] = n_params
        mi["num_parameters_millions"] = round(n_params / 1e6, 1)
        mi["config"] = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in config.__dict__.items() if not k.startswith("_")
        }

    def record_epoch(self, *, train_loss, val_loss, lr, epoch_time,
                     gpu_allocated, gpu_reserved, gpu_nvidia_smi,
                     data_time, forward_time, backward_time,
                     grad_norm_avg, grad_norm_max, grad_clip_count, nan_inf_count,
                     ema_divergence, loss_buckets, weight_norm, weight_update_ratio):
        pm = self.stats["per_epoch_metrics"]
        pm["train_losses"].append(train_loss)
        pm["val_losses"].append(val_loss)
        pm["learning_rates"].append(lr)
        pm["epoch_times_seconds"].append(epoch_time)
        pm["timestamps"].append(datetime.now().isoformat())
        pm["gpu_memory_allocated_gb"].append(gpu_allocated)
        pm["gpu_memory_reserved_gb"].append(gpu_reserved)
        pm["gpu_memory_nvidia_smi_gb"].append(gpu_nvidia_smi)
        pm["data_loading_time_sec"].append(data_time)
        pm["forward_pass_time_sec"].append(forward_time)
        pm["backward_pass_time_sec"].append(backward_time)
        pm["gradient_norms"].append(grad_norm_avg)
        pm["gradient_norms_max"].append(grad_norm_max)
        pm["gradient_clip_counts"].append(grad_clip_count)
        pm["nan_inf_counts"].append(nan_inf_count)

        # loss spike if this epoch jumps >50% above the previous one
        spike = False
        if len(pm["train_losses"]) >= 2:
            prev = pm["train_losses"][-2]
            spike = prev > 0 and train_loss > prev * 1.5
        pm["loss_spike_flags"].append(spike)

        pm["ema_divergences"].append(ema_divergence)
        pm["loss_bucket_0_50"].append(loss_buckets.get(0, 0.0))
        pm["loss_bucket_50_100"].append(loss_buckets.get(1, 0.0))
        pm["loss_bucket_100_150"].append(loss_buckets.get(2, 0.0))
        pm["loss_bucket_150_200"].append(loss_buckets.get(3, 0.0))
        pm["loss_bucket_200_250"].append(loss_buckets.get(4, 0.0))
        pm["weight_norms"].append(weight_norm)
        pm["weight_update_ratios"].append(weight_update_ratio)

    def finalize(self):
        pm = self.stats["per_epoch_metrics"]
        times = pm["epoch_times_seconds"]
        if not times:
            return

        ts = self.stats["timing_summary"]
        ts["average_epoch_time_sec"] = round(sum(times) / len(times), 2)
        ts["fastest_epoch_time_sec"] = round(min(times), 2)
        ts["slowest_epoch_time_sec"] = round(max(times), 2)
        ts["median_epoch_time_sec"] = round(sorted(times)[len(times) // 2], 2)
        ts["total_data_loading_time_sec"] = round(sum(pm["data_loading_time_sec"]), 2)
        ts["total_forward_pass_time_sec"] = round(sum(pm["forward_pass_time_sec"]), 2)
        ts["total_backward_pass_time_sec"] = round(sum(pm["backward_pass_time_sec"]), 2)

        ti = self.stats["training_info"]
        total_sec = sum(times)
        ti["total_time_hours"] = round(total_sec / 3600, 2)
        h, rem = divmod(int(total_sec), 3600)
        m, s = divmod(rem, 60)
        ti["total_time_formatted"] = f"{h}h {m}m {s}s"
        ti["total_gpu_hours"] = ti["total_time_hours"]

        nv = pm["gpu_memory_nvidia_smi_gb"]
        if any(v is not None for v in nv):
            ti["peak_memory_gb"] = max(v for v in nv if v is not None)
        elif pm["gpu_memory_reserved_gb"]:
            ti["peak_memory_gb"] = max(pm["gpu_memory_reserved_gb"])

        cv = self.stats["convergence"]
        val_losses = pm["val_losses"]
        best_idx = val_losses.index(min(val_losses))
        cv["best_epoch"] = best_idx + 1
        cv["best_val_loss"] = val_losses[best_idx]
        cv["final_train_loss"] = pm["train_losses"][-1]
        cv["final_val_loss"] = val_losses[-1]

        # epoch where val loss first comes within 5% of best
        threshold = cv["best_val_loss"] * 1.05
        for i, vl in enumerate(val_losses):
            if vl <= threshold:
                cv["convergence_epoch_95pct"] = i + 1
                break

        # earliest epoch with 10 consecutive monotonic increases in val loss
        for i in range(10, len(val_losses)):
            window = val_losses[i - 10:i]
            if all(window[j] < window[j + 1] for j in range(len(window) - 1)):
                cv["overfitting_epoch"] = i - 9
                break

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
