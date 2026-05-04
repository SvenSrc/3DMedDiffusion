import os
from pathlib import Path

import numpy as np
import nibabel as nib


def load_nifti(path):
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    # some BraTS files come in as 4D with a single timepoint
    if vol.ndim == 4:
        vol = vol[..., 0]
    meta = {
        "affine": nii.affine,
        "spacing": tuple(float(s) for s in nii.header.get_zooms()[:3]),
    }
    return vol, meta


def save_nifti(volume, path, affine=None, spacing=None):
    if affine is None:
        affine = np.eye(4)
        if spacing is not None:
            affine[:3, :3] = np.diag(spacing)

    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    if spacing is not None:
        nii.header.set_zooms(spacing)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nib.save(nii, path)


def find_t1n_files(root):
    # BraTS folder layout: root / BraTS-GLI-XXXXX-NNN / BraTS-GLI-XXXXX-NNN-t1n.nii.gz
    
    files = []
    for path in Path(root).glob("**/*-t1n.nii.gz"):
        subject_id = path.stem.replace("-t1n.nii", "")
        files.append((subject_id, str(path)))
    files.sort()
    return files
