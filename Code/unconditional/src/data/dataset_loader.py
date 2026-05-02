import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class LoadDataset(Dataset):

    def __init__(self, data_path):
        self.files = sorted(Path(data_path).glob("*.nii.gz"))
        assert len(self.files) > 0, f"no .nii.gz files in {data_path}"

        # peek at the first volume so we know the shape upfront
        sample = nib.load(self.files[0]).get_fdata()
        self.volume_shape = sample.shape
        print(f"loaded {len(self.files)} volumes, shape={self.volume_shape}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = nib.load(self.files[idx]).get_fdata().astype(np.float32)
        return torch.from_numpy(vol).unsqueeze(0)
