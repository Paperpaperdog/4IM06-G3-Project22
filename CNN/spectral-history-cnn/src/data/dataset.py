from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, load_metadata: bool = False):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.spectra = np.load(self.processed_dir / f"{split}_spectra.npy")
        self.labels = np.load(self.processed_dir / f"{split}_labels.npy")
        if len(self.spectra) != len(self.labels):
            raise ValueError(f"Spectra/label length mismatch for split '{split}'.")
        self.metadata: Optional[pd.DataFrame] = None
        metadata_path = self.processed_dir / f"{split}_metadata.csv"
        if load_metadata and metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.spectra[idx].astype(np.float32))
        y = int(self.labels[idx])
        return x, y, int(idx)
