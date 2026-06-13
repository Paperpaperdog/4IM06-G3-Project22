# Spectral Mask Classifier for JPEG vs Downsampling Detection

This project studies whether JPEG compression and downsampling histories leave distinguishable traces in residual Fourier spectra.

Version 1 implements a lightweight spectral-mask baseline for six-class image processing history detection:

- original
- JPEG quality 80
- downsample_x2
- downsample_x4
- downsample_x8
- downsample_x16

The Version 1 pipeline uses Y-channel TV residuals, real FFT log magnitude, soft DC suppression, one learnable mask per class, one learnable reference spectrum per class, and multi-class cross entropy.

The first scientific question is not only whether accuracy is high, but whether JPEG, downsample_x8, and downsample_x16 learn overlapping or separable frequency masks.

## Version 1 Scope

Version 1 intentionally includes only the basic experiment:

- RAISE-1k source images from `../RAISE_1k.csv`, using the `TIFF` URL column.
- Image-level train/val/test split: 700/150/150.
- Ten random 512x512 crops per source image.
- Six classes generated from each crop.
- JPEG fixed at quality 80.
- Downsampling fixed to bicubic interpolation.
- TV denoising residual.
- `torch.fft.rfft2` with vertical `fftshift`.
- Soft DC suppression before log magnitude.
- Single real-valued spectral channel: `log(1 + abs(F))`.
- Multi-class learnable mask/reference classifier.
- Cross entropy training.

Version 1 intentionally excludes:

- CNN backbones (take too long time and ressource to train).
- Positional encoding.
- phase, real, or imaginary spectrum channels.
- DDP.
- residual/interpolation/JPEG-quality ablations.
- mixed operation classes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

By default the preparation script reads the repository-level `../RAISE_1k.csv` file and uses its `TIFF` URL column. Downloaded TIFF files are cached under `data/raw/raise_tiff/`.

```bash
bash scripts/run_v1_prepare.sh
```

For a smoke test:

```bash
export PYTHONPATH=.

python src/data/split_raise.py \
  --input-csv ../RAISE_1k.csv \
  --url-column TIFF \
  --output-json data/splits/debug_raise_split_seed123.json \
  --train 700 --val 150 --test 150 --seed 123

python src/data/preprocess_spectra.py \
  --split-json data/splits/debug_raise_split_seed123.json \
  --output-dir data/processed/debug_v1 \
  --download-dir data/raw/raise_tiff \
  --patch-size 512 \
  --patches-per-image 2 \
  --limit-images 20 \
  --jpeg-quality 80 \
  --downsample-factors 2 4 8 16 \
  --interpolation bicubic \
  --residual tv \
  --tv-weight 0.08 \
  --tv-max-iter 10 \
  --dc-sigma-bins 3.0 \
  --seed 123 \
  --dtype float16
```

For the debug cache, change `data_dir` in `configs/v1_tv_rfft_mask.yaml` to `data/processed/debug_v1` and reduce epochs if needed.

## Train

```bash
bash scripts/run_v1_train.sh
```

## Evaluate and Visualize

```bash
bash scripts/run_v1_eval.sh
```

Outputs are written to `outputs/v1_tv_rfft_mask/`.

## Outputs

Version 1 produces:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `logs/train_log.csv`
- `metrics.json`
- `predictions_test.csv`
- `figures/confusion_matrix.png`
- `figures/loss_curves.png`
- `figures/accuracy_curves.png`
- `figures/masks/`
- `figures/references/`
- `figures/mask_overlap.png`
- `figures/mean_spectra/`

The most important confusion pairs to inspect are:

- JPEG vs downsample_x8
- JPEG vs downsample_x16
- downsample_x4 vs downsample_x8
- downsample_x8 vs downsample_x16
- original vs JPEG
- original vs downsample_x2

## Implementation Notes

The code is structured so later versions can extend the pipeline without rewriting Version 1:

- `src/processing/transforms.py` contains image-history operations such as JPEG and downsample-upsample.
- `src/processing/residuals.py` contains residual extraction.
- `src/processing/spectrum.py` contains Fourier feature construction.
- `src/data/preprocess_spectra.py` owns cached dataset generation.
- `src/data/dataset.py` loads cached spectra through memory mapping.
- `src/models/spectral_mask_classifier.py` contains the Version 1 mask/reference model.
- `src/train.py`, `src/evaluate.py`, and `src/visualize.py` are model-agnostic enough to reuse with small changes.

Version 1 still deliberately validates some assumptions in code, for example `--residual tv` and downsample factors `2 4 8 16`. These checks keep the first experiment reproducible. Future versions should relax those checks only when the corresponding experiment is implemented.

## Roadmap

### Version 2: Stronger Non-CNN Forensics

Keep the same mask/reference model and expand the data generation and analysis.

Add:

- JPEG quality mixture: `70, 75, 80, 85, 90, 95`.
- Interpolation mixture: bilinear, bicubic, Lanczos.
- Mixed operation classes:
  - JPEG_then_downsample_x2/x4/x8/x16
  - downsample_then_JPEG_x2/x4/x8/x16
- crop/grid shift experiments.
- optional mask L1 regularization sweep: `0, 1e-5, 1e-4`.

Add ablations after Version 1 is stable:

- TV residual vs rank residual vs Laplacian residual.
- with DC suppression vs without DC suppression.
- rFFT vs full FFT only if there is a concrete reason to test redundancy.

Use the same image-level split whenever possible.

### Version 3: Lightweight CNN With Frequency Positional Encoding

Add a shallow real-valued CNN that consumes spectral features.

Base input:

- `log(1 + abs(F))`

Optional frequency-coordinate channels:

- `U`
- `V`
- radius `r`
- `cos(theta)`
- `sin(theta)`
- sinusoidal coordinate bands for periods `1, 2, 4, 8, 16, 32, 64`

Use Conv-BN-ReLU blocks, global average pooling, and a linear classifier.

Do not use a complex-valued CNN. Phase, real, and imaginary channels should remain separate later ablations, not part of the first CNN baseline.

### Version 4: Hybrid Fourier Mask + JPEG DCT Evidence

Add a JPEG-specific 8x8 DCT quantization evidence branch.

Possible DCT features:

- number of detected quantization table entries.
- minimum NFA or log-NFA.
- mean confidence over detected entries.
- estimated quantization value statistics.
- evidence strength on the JPEG 8x8 grid.

Combine:

- Fourier mask/reference scores.
- DCT quantization features.
- final classifier.

This version targets the key ambiguity where JPEG and downsampling both create 8/16-like spectral traces, but JPEG should also leave block-DCT quantization evidence.

### Version 5: Full Evaluation

Run systematic comparisons:

- residual type.
- DC suppression.
- fixed vs mixed JPEG quality.
- interpolation type.
- operation order.
- mask/reference vs CNN vs hybrid.
- DCT branch vs no DCT branch.
- cross-dataset generalization if another high-quality source dataset is available.
