import torch
from torch.utils.data import Dataset


class FlipAugment(Dataset):
    # random left-right flip

    def __init__(self, base, p=0.5):
        self.base = base
        self.p = p

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]
        if torch.rand(1).item() < self.p:
            x = torch.flip(x, dims=[-1])   # last spatial axis = W
        return x
