import os

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from nifti import load_nifti, save_nifti, find_t1n_files, find_seg_files


# Pipeline: percentile clip -> crop foreground -> pad to cubic -> resize -> normalize to [-1, 1].
# This follows the preprocessing method of Khader et al. 2023 


def percentile_clip(volume, lower=0.5, upper=99.5):
    foreground = volume[volume > 0]
    if foreground.size == 0:
        return volume
    lo, hi = np.percentile(foreground, [lower, upper])
    out = np.clip(volume, lo, hi)
    out[volume == 0] = 0
    return out


def foreground_bbox(volume, threshold=0.0, margin=2):
    coords = np.array(np.where(volume > threshold))
    if coords.size == 0:
        return tuple(slice(0, s) for s in volume.shape)
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    return tuple(
        slice(max(0, lo - margin), min(size, hi + margin))
        for lo, hi, size in zip(mins, maxs, volume.shape)
    )


def crop_foreground(volume, threshold=0.0, margin=2):
    return volume[foreground_bbox(volume, threshold, margin)]


def pad_to_cubic(volume, value=0.0):
    max_dim = max(volume.shape)
    pad_width = [
        (diff // 2, diff - diff // 2)
        for diff in (max_dim - s for s in volume.shape)
    ]
    return np.pad(volume, pad_width, mode="constant", constant_values=value)


def resize_volume(volume, target_size, order=1):
    factors = tuple(t / s for t, s in zip(target_size, volume.shape))
    return ndimage.zoom(volume, factors, order=order)


def normalize_to_range(volume, lo=-1.0, hi=1.0):
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin) * (hi - lo) + lo


def preprocess_volume(volume, target_size, lower=0.5, upper=99.5):
    volume = percentile_clip(volume, lower, upper)
    volume = crop_foreground(volume, threshold=0.0, margin=2)
    volume = pad_to_cubic(volume)
    volume = resize_volume(volume, target_size, order=1)
    volume = normalize_to_range(volume, -1.0, 1.0)
    return volume


def preprocess_seg(t1n, seg, target_size, lower=0.5, upper=99.5):
    # crop seg with the bbox computed from t1n so the two stay aligned
    t1n_clipped = percentile_clip(t1n, lower, upper)
    bbox = foreground_bbox(t1n_clipped)
    seg = seg[bbox]
    seg = pad_to_cubic(seg)
    seg = resize_volume(seg, target_size, order=0)   # nearest-neighbor preserves labels
    return seg


def process_brats_dataset(input_dir, output_dir, target_size, lower=0.5, upper=99.5):
    os.makedirs(output_dir, exist_ok=True)
    files = find_t1n_files(input_dir)
    print(f"found {len(files)} T1N volumes, target size {target_size}")

    written = []
    for subject_id, path in tqdm(files, desc="preprocessing"):
        try:
            vol, _ = load_nifti(path)
            vol = preprocess_volume(vol, target_size, lower, upper)
            out_path = os.path.join(output_dir, f"{subject_id}.nii.gz")
            save_nifti(vol, out_path, spacing=(1.0, 1.0, 1.0))
            written.append(out_path)
        except Exception as e:
            print(f"  skipped {subject_id}: {e}")

    print(f"done: {len(written)}/{len(files)} written to {output_dir}")
    return written


def process_brats_seg_dataset(input_dir, output_dir, target_size, lower=0.5, upper=99.5):
    os.makedirs(output_dir, exist_ok=True)
    pairs = find_seg_files(input_dir)
    print(f"found {len(pairs)} (T1N, seg) pairs, target size {target_size}")

    written = []
    for subject_id, t1n_path, seg_path in tqdm(pairs, desc="preprocessing seg"):
        try:
            t1n, _ = load_nifti(t1n_path)
            seg, _ = load_nifti(seg_path)
            seg = preprocess_seg(t1n, seg, target_size, lower, upper)
            out_path = os.path.join(output_dir, f"{subject_id}.nii.gz")
            save_nifti(seg, out_path, spacing=(1.0, 1.0, 1.0), dtype=np.int16)
            written.append(out_path)
        except Exception as e:
            print(f"  skipped {subject_id}: {e}")

    print(f"done: {len(written)}/{len(pairs)} written to {output_dir}")
    return written
