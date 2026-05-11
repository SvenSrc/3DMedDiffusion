import os
from pathlib import Path

import numpy as np
import nibabel as nib


def load_nifti(path):
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    meta = {
        "affine": nii.affine,
        "spacing": tuple(float(s) for s in nii.header.get_zooms()[:3]),
    }
    return vol, meta


def save_nifti(volume, path, affine=None, spacing=None, dtype=np.float32):
    if affine is None:
        affine = np.eye(4)
        if spacing is not None:
            affine[:3, :3] = np.diag(spacing)

    nii = nib.Nifti1Image(volume.astype(dtype), affine)
    if spacing is not None:
        nii.header.set_zooms(spacing)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nib.save(nii, path)


def find_t1n_files(root):
    files = []
    for path in Path(root).glob("**/*-t1n.nii.gz"):
        subject_id = path.stem.replace("-t1n.nii", "")
        files.append((subject_id, str(path)))
    files.sort()
    return files


def find_seg_files(root):
    # returns (subject_id, t1n_path, seg_path) — only subjects that have both files
    pairs = []
    for seg_path in Path(root).glob("**/*-seg.nii.gz"):
        subject_id = seg_path.stem.replace("-seg.nii", "")
        t1n_path = seg_path.parent / f"{subject_id}-t1n.nii.gz"
        if t1n_path.exists():
            pairs.append((subject_id, str(t1n_path), str(seg_path)))
    pairs.sort()
    return pairs
