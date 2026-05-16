import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class BraTSCondDataset(Dataset):
    # Loads paired (T1N volume, segmentation mask) for conditional training.
    # T1N is already in [-1, 1] from preprocessing; seg labels {0..K-1}
    # are remapped to [-1, 1] on the fly so the cond stream is on the same
    # scale as the noised input.

    def __init__(self, data_path, seg_path, num_seg_classes=4):
        self.num_seg_classes = num_seg_classes

        vol_files = {p.stem: p for p in Path(data_path).glob("*.nii.gz")}
        seg_files = {p.stem: p for p in Path(seg_path).glob("*.nii.gz")}

        common = sorted(set(vol_files) & set(seg_files))
        assert len(common) > 0, f"no overlapping subjects between {data_path} and {seg_path}"

        self.vol_files = [vol_files[s] for s in common]
        self.seg_files = [seg_files[s] for s in common]

        # peek at the first pair for shape + label info
        sample_vol = nib.load(self.vol_files[0]).get_fdata()
        sample_seg = nib.load(self.seg_files[0]).get_fdata()
        self.volume_shape = sample_vol.shape
        print(f"loaded {len(self.vol_files)} (vol, seg) pairs, shape={self.volume_shape}, "
              f"labels={np.unique(sample_seg).astype(int).tolist()}")

    def __len__(self):
        return len(self.vol_files)

    def __getitem__(self, idx):
        vol = nib.load(self.vol_files[idx]).get_fdata().astype(np.float32)
        seg = nib.load(self.seg_files[idx]).get_fdata().astype(np.float32)
        # discrete labels {0..K-1} -> continuous [-1, 1]
        seg = seg / (self.num_seg_classes - 1) * 2.0 - 1.0
        return (torch.from_numpy(vol).unsqueeze(0),
                torch.from_numpy(seg).unsqueeze(0))
