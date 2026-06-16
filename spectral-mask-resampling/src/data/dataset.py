from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str):
        data_dir = Path(data_dir)
        self.spectra = np.load(data_dir / f"{split}_spectra.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.spectra[index], dtype=np.float32))
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y
