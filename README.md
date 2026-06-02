# 4IM06-G3-Project22

This repository contains a clean-room implementation of the spectral
correlation resampling detector from `resampling_detection (1).pdf`, plus a
controlled experiment pipeline for estimating possible original image sizes.

The current research question is not only whether an image was resampled, but
whether the spectral traces can help rank candidate original sizes. For example,
if a target image has size `C = 384` and a strong peak appears at `d = 128`,
then several original sizes can explain the same peak:

```text
N mod C = 128
N mod C = C - 128 = 256
```

This creates candidates such as `128, 256, 512, 640, 896, 1024`. The code
therefore treats original-size recovery as a candidate generation and ranking
problem, not as a unique inverse problem.

## Files

- `resampling_core.py`: core detector implementation:
  - TV residual extraction;
  - centered Fourier spectrum computation;
  - complex Pearson correlation between shifted spectrum patches;
  - a-contrario NFA computation.
- `candidate_estimation.py`: candidate original-size generation and ranking
  from detected NFA peaks.
- `demo_resampling_detection.py`: small baseline demo on either
  `skimage.data.camera` or a user-provided image.
- `synthesize_controlled_resampling_dataset.py`: batch synthesis of controlled
  resampling data from RAISE TIFF images.
- `run_detector_on_synthesized_dataset.py`: batch detector runner for an
  already synthesized dataset.
- `run_controlled_resampling_experiments.py`: older combined script that
  synthesizes and detects in one pass; useful for small experiments.
- `RAISE_1k.csv`: metadata for the RAISE subset, including TIFF download links.
- `SUIVI.md`: project meeting notes and progress tracking.

## Baseline Detector

Run the minimal detector demo:

```bash
.venv/bin/python demo_resampling_detection.py
```

With no argument, the demo resamples `skimage.data.camera` from `512 x 512` to
`384 x 384`. The expected anomalous distances are:

```text
512 mod 384 = 128
-512 mod 384 = 256
```

Run it on a local image:

```bash
.venv/bin/python demo_resampling_detection.py path/to/image.png
```

The detector writes NFA curves and CSV values under:

```text
test_results/resampling_detection/
```

## Method Summary

1. Convert the image to grayscale floating point values.
2. Use TV denoising to estimate the smooth image component.
3. Compute the residual `image - TV(image)`.
4. Transform the residual with a centered 2D FFT.
5. Split the spectrum into non-overlapping patches.
6. For each distance `d`, compare every patch with the circularly shifted
   spectrum patch using complex Pearson correlation.
7. Count how many patches have a local maximum at each distance.
8. Convert these counts to an NFA with a binomial a-contrario model.
9. Use low-NFA peak distances to generate candidate original sizes.

## Candidate Size Estimation

For one axis with current size `C` and detected peak distance `d`, the candidate
generator enumerates:

```text
N = k*C + d
N = k*C + (C-d)
```

within a reasonable scale range. Each candidate predicts one or two peak
positions, and `candidate_estimation.py` ranks candidates by the observed
`-log10(NFA)` support at those predicted positions.

Important limitation: candidates with the same residue modulo `C` can receive
the same score. This is a real identifiability issue, not just an implementation
bug. The controlled dataset is designed to measure this ambiguity.

## Controlled Dataset

The synthesized bicubic RAISE dataset uses:

```text
images: 100 RAISE TIFF images
final interpolation: bicubic
target sizes: 256, 384, 512
designed peaks per target: 3
source sizes per target/peak group: 5
total target images: 100 * 3 * 3 * 5 = 4500
```

Designed peak groups:

```text
target 256: peaks 64, 85, 96
target 384: peaks 96, 128, 160
target 512: peaks 128, 171, 192
```

For each `(target_size, designed_peak)`, source sizes are generated from:

```text
k*C + d
k*C + (C-d)
```

with `k = 0, 1, 2`, filtering out very small source sizes and keeping 5 sizes
per peak group.

Synthesis command with TIFF download from `RAISE_1k.csv`:

```bash
.venv/bin/python synthesize_controlled_resampling_dataset.py 100 --download
```

If TIFF files are already local, omit `--download`:

```bash
.venv/bin/python synthesize_controlled_resampling_dataset.py 100 \
  --image-dir path/to/local/tiff_images
```

## Batch Detection

After synthesis, run the detector on all target images:

```bash
.venv/bin/python run_detector_on_synthesized_dataset.py \
  test_results/controlled_resampling_dataset_bicubic_raise100/metadata.csv
```

For a small check:

```bash
.venv/bin/python run_detector_on_synthesized_dataset.py \
  test_results/controlled_resampling_dataset_bicubic_raise100/metadata.csv \
  --limit 10
```

Each case directory receives:

```text
target.png
spectrum.png
nfa_curves.png
vertical_nfa.csv
horizontal_nfa.csv
vertical_peaks.csv
horizontal_peaks.csv
vertical_candidates.csv
horizontal_candidates.csv
```

The batch summary is:

```text
test_results/controlled_resampling_dataset_bicubic_raise100/detection_summary.csv
```

## Result Directories

- `test_results/controlled_resampling_dataset_bicubic_raise100/`: main
  controlled bicubic dataset and detection results.
  - `metadata.csv`: one row per synthesized target image.
  - `detection_summary.csv`: one vertical and one horizontal detector summary
    row per target image.
  - `references/`: local reference PNG images converted from TIFF.
  - `controlled_sources/`: source-size images before the final target resize.
  - `cases/`: per-target outputs, including target images, spectra, NFA curves,
    and candidate CSV files.
- `test_results/raise_tiff_downloads/`: local TIFF download cache. This is
  raw data cache, not a result to commit.
- `test_results/resampling_detection/`: baseline demo detector outputs.
- `test_results/controlled_resampling_raise1_sample/`: small earlier sample
  experiment with multiple interpolation methods.

Large generated images are intentionally ignored by git because this project is
not using Git LFS. The lightweight CSV summaries are suitable for the repository;
full images should be regenerated locally or shared through an external dataset
archive if needed.
