from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse processed spectra by remapping/subsetting classes.")
    parser.add_argument("--src-processed-dir", required=True)
    parser.add_argument("--dst-processed-dir", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    args = parser.parse_args()

    src_dir = Path(args.src_processed_dir)
    dst_dir = Path(args.dst_processed_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(args.classes)
    class_to_new_id = {name: i for i, name in enumerate(class_names)}

    for split in ("train", "val", "test"):
        spectra_path = src_dir / f"{split}_spectra.npy"
        labels_path = src_dir / f"{split}_labels.npy"
        meta_path = src_dir / f"{split}_metadata.csv"

        spectra = np.load(spectra_path)
        labels = np.load(labels_path)
        meta = pd.read_csv(meta_path)

        if len(spectra) != len(labels) or len(labels) != len(meta):
            raise ValueError(
                f"Length mismatch in split={split}: spectra={len(spectra)} labels={len(labels)} meta={len(meta)}"
            )

        keep_mask = meta["class_name"].isin(class_names).to_numpy()
        kept_spectra = spectra[keep_mask]
        kept_meta = meta.loc[keep_mask].reset_index(drop=True).copy()
        kept_meta["class_id"] = kept_meta["class_name"].map(class_to_new_id).astype(np.int64)
        kept_labels = kept_meta["class_id"].to_numpy(dtype=np.int64)

        np.save(dst_dir / f"{split}_spectra.npy", kept_spectra)
        np.save(dst_dir / f"{split}_labels.npy", kept_labels)
        kept_meta.to_csv(dst_dir / f"{split}_metadata.csv", index=False)
        print(f"{split}: kept={len(kept_labels)} shape={kept_spectra.shape}")

    with (dst_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Wrote remapped dataset to {dst_dir}")


if __name__ == "__main__":
    main()
