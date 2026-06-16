# Spectral Mask Classifier for JPEG vs Downsampling Ambiguity

This project studies whether JPEG compression and downsampling leave distinguishable traces in residual Fourier spectra, especially when both operations create similar 8/16-period spectral artifacts.

The main scientific question is not ordinary downsampling-factor classification. The goal is to test whether a Fourier-only learned spectral mask can separate JPEG-specific traces from downsampling-specific traces under a controlled ambiguity setting.

Version 1 implements a lightweight Fourier-only ambiguity test with four classes:

- `original`
- `JPEG_Q80`
- `downsample_x8`
- `downsample_x16`

All samples may have different observed image sizes. Their residual spectra are mapped to a common normalized frequency grid before entering the mask/reference classifier. This is frequency-domain resampling, not image-domain upsampling.

## Version 1 Scope

Version 1 intentionally includes only the basic experiment:

- RAISE-1k source images from `../RAISE_1k.csv`, using the `TIFF` URL column.
- Image-level train/val/test split.
- Four-class ambiguity experiment: `original`, `JPEG_Q80`, `downsample_x8`, `downsample_x16`.
- Observed image sizes: `128, 96, 64, 48, 32`.
- For each observed size:
  - `original`: directly crop an `observed_size x observed_size` patch.
  - `JPEG_Q80`: directly crop an `observed_size x observed_size` patch, then JPEG-compress/decode at quality 80.
  - `downsample_x8`: crop a source patch of size `observed_size * 8`, then resize it to `observed_size`.
  - `downsample_x16`: crop a source patch of size `observed_size * 16`, then resize it to `observed_size`, only when the source patch fits the source image.
- Downsampling interpolation fixed to bicubic.
- Y-channel TV denoising residual.
- `torch.fft.rfft2` with vertical `fftshift`.
- Single real-valued spectral channel: `log(1 + abs(F))`.
- Map every variable-size spectrum to a common normalized frequency grid of size `512 x 257`.
- Soft DC suppression on the common frequency grid.
- Multi-class learnable mask/reference classifier.
- One learnable mask `M_k` and one learnable reference spectrum `R_k` per class.
- Multi-class cross entropy training.

Version 1 intentionally excludes:

- image-domain downsample-then-upsample classes.
- `downsample_x2` and `downsample_x4`, because the first experiment focuses on the hardest JPEG-vs-8/16 ambiguity.
- CNN backbones.
- positional encoding.
- phase, real, or imaginary spectrum channels.
- DDP.
- residual/interpolation/JPEG-quality ablations.
- mixed operation classes such as `JPEG_then_downsample`.
- JPEG DCT quantization evidence branch.

## Scientific Interpretation of Version 1

JPEG compression often introduces 8x8 block-DCT artifacts. In Fourier spectra, these can appear as periodic or grid-like traces around period 8 and sometimes period 16. Downsampling by factors 8 and 16 can also create frequency-domain structures related to the same periods. Version 1 asks:

> Can a learned Fourier spectral mask distinguish JPEG compression from downsampling x8/x16 when their dominant periodic traces may overlap?

The experiment should not be judged only by overall accuracy. The most important outputs are the confusion matrix, per-observed-size accuracy, learned class masks, and mask-overlap matrix. If JPEG and downsampling x8/x16 are heavily confused, this is still a useful result: it supports the motivation for adding JPEG-specific DCT quantization evidence in a later version.

## Data Generation Matrix

Recommended Version 1 matrix:

```text
Classes:
0 original
1 JPEG_Q80
2 downsample_x8
3 downsample_x16

Observed sizes:
128, 96, 64, 48, 32

For each observed size o:
original:       crop o x o directly
JPEG_Q80:       crop o x o directly, JPEG-compress/decode at Q=80
downsample_x8:  crop (8o) x (8o), resize to o x o
downsample_x16: crop (16o) x (16o), resize to o x o if it fits
```

If some RAISE images are too small for a requested source patch, skip that sample and continue sampling from other images. Keep the class/size distribution as balanced as possible.

Recommended first target:

```text
samples per class per observed size: 1000
4 classes x 5 sizes x 1000 = 20k samples
```

Larger target if data generation is fast enough:

```text
samples per class per observed size: 2000
4 classes x 5 sizes x 2000 = 40k samples
```

## Frequency Pipeline

For every generated sample:

```text
RGB image
-> Y channel
-> TV residual
-> rFFT2
-> vertical fftshift
-> log(1 + abs(F))
-> map to common normalized frequency grid, 512 x 257
-> DC suppression on the common grid
-> spectral mask/reference classifier
```

The normalized frequency grid uses cycles per pixel. The horizontal rFFT coordinate is in `[0, 0.5]`, and the vertical coordinate is approximately in `[-0.5, 0.5)`. This lets spectra from different image sizes be compared in a common frequency coordinate system.

Important: the variable-size images are not resized back to `512 x 512` in image space. Only their log-rFFT spectra are interpolated to the common frequency grid.

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

Expected `scripts/run_v1_prepare.sh`:

```bash
#!/usr/bin/env bash
set -e

export PYTHONPATH=.

python src/data/split_raise.py \
  --input-csv ../RAISE_1k.csv \
  --url-column TIFF \
  --output-json data/splits/raise_split_seed123.json \
  --train 700 --val 150 --test 150 --seed 123

python src/data/preprocess_spectra.py \
  --split-json data/splits/raise_split_seed123.json \
  --output-dir data/processed/v1_fourier_ambiguity \
  --download-dir data/raw/raise_tiff \
  --classes original JPEG_Q80 downsample_x8 downsample_x16 \
  --observed-sizes 128 96 64 48 32 \
  --samples-per-class-per-size 1000 \
  --jpeg-quality 80 \
  --downsample-factors 8 16 \
  --interpolation bicubic \
  --residual tv \
  --tv-weight 0.08 \
  --tv-max-iter 30 \
  --target-spectrum-height 512 \
  --target-spectrum-width-rfft 257 \
  --dc-sigma-bins 3.0 \
  --seed 123 \
  --dtype float16
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
  --output-dir data/processed/debug_v1_fourier_ambiguity \
  --download-dir data/raw/raise_tiff \
  --classes original JPEG_Q80 downsample_x8 downsample_x16 \
  --observed-sizes 128 96 64 48 32 \
  --samples-per-class-per-size 20 \
  --limit-images 20 \
  --jpeg-quality 80 \
  --downsample-factors 8 16 \
  --interpolation bicubic \
  --residual tv \
  --tv-weight 0.08 \
  --tv-max-iter 10 \
  --target-spectrum-height 512 \
  --target-spectrum-width-rfft 257 \
  --dc-sigma-bins 3.0 \
  --seed 123 \
  --dtype float16
```

For the debug cache, change `data_dir` in `configs/v1_fourier_ambiguity_mask_clean.yaml` to `data/processed/debug_v1_fourier_ambiguity` and reduce epochs if needed.

## Train

```bash
bash scripts/run_v1_train.sh
```

Expected `scripts/run_v1_train.sh`:

```bash
#!/usr/bin/env bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

python src/train.py \
  --config configs/v1_fourier_ambiguity_mask_clean.yaml
```

Expected config:

```yaml
experiment_name: v1_fourier_ambiguity_mask_clean
data_dir: data/processed/v1_fourier_ambiguity
output_dir: outputs/v1_fourier_ambiguity_mask_clean

num_classes: 4
class_names:
  - original
  - JPEG_Q80
  - downsample_x8
  - downsample_x16

observed_sizes:
  - 128
  - 96
  - 64
  - 48
  - 32

spectrum:
  channels: 1
  height: 512
  width_rfft: 257

model:
  type: spectral_mask_classifier
  init_mask_logits: 0.0
  init_reference_std: 0.02
  lambda_mask_l1: 0.0

training:
  device: cuda
  batch_size: 64
  epochs: 30
  optimizer: AdamW
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  seed: 123
  num_workers: 4
  pin_memory: true
  save_best_by: val_loss
```

## Evaluate and Visualize

```bash
bash scripts/run_v1_eval.sh
```

Expected `scripts/run_v1_eval.sh`:

```bash
#!/usr/bin/env bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

python src/evaluate.py \
  --config configs/v1_fourier_ambiguity_mask_clean.yaml \
  --checkpoint outputs/v1_fourier_ambiguity_mask_clean/checkpoints/best.pt \
  --split test

python src/visualize.py \
  --config configs/v1_fourier_ambiguity_mask_clean.yaml \
  --checkpoint outputs/v1_fourier_ambiguity_mask_clean/checkpoints/best.pt
```

Outputs are written to `outputs/v1_fourier_ambiguity_mask_clean/`.

## Outputs

Version 1 produces:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `logs/train_log.csv`
- `metrics.json`
- `predictions_test.csv`
- `figures/confusion_matrix.png`
- `figures/confusion_matrix_by_observed_size/`
- `figures/loss_curves.png`
- `figures/accuracy_curves.png`
- `figures/accuracy_by_observed_size.png`
- `figures/masks/`
- `figures/references/`
- `figures/mask_overlap.png`
- `figures/mean_spectra/`

The most important confusion pairs to inspect are:

- `JPEG_Q80 -> downsample_x8`
- `JPEG_Q80 -> downsample_x16`
- `downsample_x8 -> JPEG_Q80`
- `downsample_x16 -> JPEG_Q80`

The most important mask overlaps to inspect are:

- `Overlap(M_JPEG_Q80, M_downsample_x8)`
- `Overlap(M_JPEG_Q80, M_downsample_x16)`
- `Overlap(M_downsample_x8, M_downsample_x16)`

The most important grouped metrics are:

- accuracy for observed size 128
- accuracy for observed size 96
- accuracy for observed size 64
- accuracy for observed size 48
- accuracy for observed size 32

If observed size 32 performs much worse, this may be caused by the very small original spectrum before interpolation, rather than by a failure of the mask model itself.

## Held-Out Observed Size Test

After the normal Version 1 run works, add a held-out observed size test without changing the model:

```text
train observed sizes: 128, 64, 32
test observed sizes: 96, 48
```

If the model still distinguishes JPEG from downsample_x8/x16 on unseen observed sizes, this supports the normalized frequency-grid design.

This can be implemented as a second split/config, for example:

```text
configs/v1_fourier_ambiguity_mask_heldout_size.yaml
outputs/v1_fourier_ambiguity_mask_heldout_size/
```

## Implementation Notes

The code should be structured so later versions can extend the pipeline without rewriting Version 1:

- `src/processing/transforms.py` contains image-history operations such as JPEG compression and direct downsampling.
- `src/processing/residuals.py` contains residual extraction.
- `src/processing/spectrum.py` contains rFFT, log magnitude, normalized frequency-grid interpolation, and DC suppression.
- `src/data/preprocess_spectra.py` owns cached dataset generation.
- `src/data/dataset.py` loads cached spectra through memory mapping.
- `src/models/spectral_mask_classifier.py` contains the Version 1 mask/reference model.
- `src/train.py`, `src/evaluate.py`, and `src/visualize.py` are model-agnostic enough to reuse with small changes.

Version 1 should deliberately validate these assumptions in code:

- classes are exactly `original`, `JPEG_Q80`, `downsample_x8`, `downsample_x16`.
- observed sizes are exactly `128, 96, 64, 48, 32`, unless a debug or held-out-size config is explicitly used.
- residual is `tv`.
- JPEG quality is `80`.
- downsample factors are `8` and `16`.
- all cached spectra have shape `[1, 512, 257]` after normalized frequency-grid mapping.

These checks keep the first experiment reproducible. Future versions should relax them only when the corresponding experiment is implemented.

## Roadmap

### Version 2: Mixed JPEG Quality

Keep the same mask/reference model and expand JPEG conditions.

Two possible designs:

Design A: JPEG remains one class.

```text
0 original
1 JPEG_mixed_QF
2 downsample_x8
3 downsample_x16
```

Use JPEG quality randomly sampled from:

```text
60, 70, 80, 90, 95
```

Design B: JPEG qualities become separate classes.

```text
0 original
1 JPEG_Q95
2 JPEG_Q80
3 JPEG_Q60
4 downsample_x8
5 downsample_x16
```

This tests whether stronger JPEG compression becomes more easily confused with downsample_x8/x16.

### Version 3: Mixed Operation Histories

Add realistic mixed processing histories:

- `JPEG_then_downsample_x8`
- `downsample_x8_then_JPEG`
- `JPEG_then_downsample_x16`
- `downsample_x16_then_JPEG`

Interpretation:

- `JPEG -> downsample`: JPEG 8x8 traces may be weakened, shifted, or blurred by the later downsampling.
- `downsample -> JPEG`: the final JPEG 8x8 grid may dominate and hide the downsampling trace.

This version directly tests whether Fourier-only masks can separate operation order.

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

### Version 5: Lightweight CNN With Frequency Positional Encoding

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

### Version 6: Full Evaluation

Run systematic comparisons:

- residual type.
- DC suppression.
- fixed vs mixed JPEG quality.
- interpolation type.
- operation order.
- mask/reference vs CNN vs hybrid.
- DCT branch vs no DCT branch.
- held-out observed size generalization.
- cross-dataset generalization if another high-quality source dataset is available.
