# ResamplingDetector

`ResamplingDetector.py` is a command-line Python tool for detecting image resampling traces using Fourier-domain spectral correlation and an a contrario Number of False Alarms (NFA) decision rule.

The detector is designed to analyze whether an image contains abnormal periodic correlations that may come from resizing / upsampling operations. It can test vertical and horizontal axes separately, suppress JPEG-related false candidates, and optionally cross-validate both axes for proportional resampling.

---

## Main Features

- Load an input image and convert it to grayscale.
- Apply optional preprocessing:
  - `rank`: rank transform for suppressing image content.
  - `tv`: Total Variation residual, useful for JPEG-compressed images.
  - `none`: no preprocessing.
- Compute the centered 2D Fourier spectrum.
- Split the spectrum into non-overlapping patches.
- Measure complex Pearson correlation between spectral patches and shifted patches.
- Count local correlation maxima for each candidate distance.
- Compute the NFA score using a binomial tail probability.
- Detect significant distances where:

```text
NFA(d) < epsilon
```

- Suppress distances likely caused by JPEG 8×8 block artifacts.
- Cross-validate vertical and horizontal detections for proportional resizing.
- Save `log10 NFA` line plots into a `results/` folder.

---

## File Structure

The script is self-contained and includes the following main components:

| Component | Purpose |
|---|---|
| `DetectionResult` | Dataclass storing detection results for one axis |
| `load_grayscale_image()` | Loads an image and converts it to grayscale |
| `rank_transform()` | Applies local rank transform preprocessing |
| `tv_residual()` | Extracts residual using TV denoising |
| `compute_spectrum()` | Computes centered Fourier spectrum |
| `extract_non_overlapping_patches()` | Splits spectrum into spectral patches |
| `detect_axis()` | Core NFA-based detection function |
| `cross_validate_proportional_resampling()` | Validates proportional resampling between two axes |
| `save_nfa_plot()` | Saves NFA curve as a line plot |
| `print_final_decision()` | Prints final image-level decision |

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install numpy pillow scipy matplotlib
```

For the `tv` preprocessing mode, install `scikit-image`:

```bash
pip install scikit-image
```

If `scikit-image` is not installed, the code automatically falls back to a simple Gaussian high-pass residual.

---

## Basic Usage

Run detection on one image:

```bash
python ResamplingDetector.py input.png
```

By default, the script uses:

```text
--preprocess rank
--patch-size 8
--r 3
--epsilon 1.0
--axis both
```

---

## Recommended Commands

### 1. For PNG or uncompressed images

```bash
python ResamplingDetector.py input.png --preprocess rank --axis both
```

### 2. For JPEG images

```bash
python ResamplingDetector.py input.jpg --preprocess tv --axis both --suppress-jpeg
```

### 3. With cross-validation between both axes

```bash
python ResamplingDetector.py input.jpg --preprocess tv --axis both --suppress-jpeg --cross-validate
```

### 4. Save NFA plots

```bash
python ResamplingDetector.py input.png --preprocess rank --axis both --plot-prefix input
```

The generated plots are saved automatically in:

```text
results/
```

Example output filenames:

```text
results/input_20260522_143012_axis0.png
results/input_20260522_143012_axis1.png
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---:|---|
| `image` | required | Path to the input image |
| `--preprocess` | `rank` | Preprocessing mode: `rank`, `tv`, or `none` |
| `--patch-size` | `8` | Square spectral patch size |
| `--r` | `3` | Local neighborhood radius for distance comparison |
| `--epsilon` | `1.0` | NFA threshold; smaller means stricter detection |
| `--axis` | `both` | Axis to test: `0`, `1`, or `both` |
| `--suppress-jpeg` | disabled | Suppress JPEG-related distances around `k*N/8` |
| `--cross-validate` | disabled | Validate proportional resampling between both axes |
| `--beta` | `0.01` | Ratio tolerance for cross-validation |
| `--plot-prefix` | `None` | If provided, saves NFA plots using this prefix |
| `--jpeg-radius` | `3` | Suppression radius around JPEG-related distances |
| `--min-distance` | `20` | Ignore very small distances and distances close to image size |

---

## Method Overview

The detection pipeline follows these steps:

### 1. Preprocessing

The image is converted to grayscale and optionally transformed before Fourier analysis.

- `rank` reduces the influence of image content by replacing each pixel with its local rank.
- `tv` removes smooth image content and keeps residual information.
- `none` directly uses the grayscale image.

### 2. Fourier Spectrum

The preprocessed image is transformed using the 2D Discrete Fourier Transform:

```python
spectrum = np.fft.fftshift(np.fft.fft2(image))
```

The spectrum is centered so that low frequencies are located in the middle.

### 3. Spectral Patch Correlation

The spectrum is divided into non-overlapping patches. For each candidate distance `d`, the spectrum is shifted along the selected axis, and the correlation between original patches and shifted patches is computed.

The code uses complex Pearson correlation:

```text
corr(x, y) = | <x, y> |
```

after mean-centering and L2-normalization.

### 4. Local Maximum Counting

For each candidate distance `d`, the detector checks whether the correlation at `d` is a local maximum within:

```text
[d - r, d + r]
```

The number of patches where `d` is the local maximum is called:

```text
k(d)
```

### 5. NFA Computation

Under the null hypothesis of no resampling, the code models:

```text
k(d) ~ Binomial(number_of_patches, 1 / (2r + 1))
```

Then it computes:

```text
NFA(d) = number_of_tests × P[X >= k(d)]
```

If:

```text
NFA(d) < epsilon
```

the distance is considered statistically significant.

### 6. JPEG Suppression

JPEG compression can create strong periodic traces around distances related to 8×8 blocks:

```text
k × N / 8,  k = 1, ..., 7
```

When `--suppress-jpeg` is enabled, the code removes these candidate distances and nearby distances from testing.

### 7. Cross-Validation

If `--cross-validate` is enabled, detections from axis 0 and axis 1 are compared using normalized distance ratios:

```text
d0 / N0 ≈ d1 / N1
```

This helps confirm proportional resampling and reduce false positives.

---

## Output Explanation

For each tested axis, the script prints:

```text
=== Detection along axis 0 / vertical ===
Axis size: ...
Detected: True / False
Best distance: ...
Best NFA: ...
Top suspicious distances:
distance    k(d)    NFA    log10(NFA)
```

### Important fields

| Field | Meaning |
|---|---|
| `distance` | Candidate spectral shift distance |
| `k(d)` | Number of patches where this distance is a local maximum |
| `NFA` | Number of False Alarms score |
| `log10(NFA)` | Logarithmic NFA value |
| `Best distance` | Distance with the smallest NFA |
| `Detected` | Whether at least one distance has `NFA < epsilon` |

---

## Final Decision

The final decision can be one of the following:

### Without cross-validation

The image is classified as resampled if at least one axis contains a distance with:

```text
NFA(d) < epsilon
```

### With cross-validation

The image is classified as resampled only if a pair of distances from the two axes passes the proportional-ratio validation.

This is usually more reliable for images that may have JPEG compression or other periodic artifacts.

---

## Practical Recommendations

For clean PNG images:

```bash
python ResamplingDetector.py image.png --preprocess rank --axis both --cross-validate
```

For JPEG images:

```bash
python ResamplingDetector.py image.jpg --preprocess tv --axis both --suppress-jpeg --cross-validate
```

For debugging or visual inspection:

```bash
python ResamplingDetector.py image.jpg --preprocess tv --axis both --suppress-jpeg --cross-validate --plot-prefix image
```

---

## Notes and Limitations

- The detector does not directly prove that an image is AI-generated. It only detects abnormal resampling-like spectral correlations.
- JPEG compression can create periodic artifacts, so `--suppress-jpeg` is recommended for JPEG inputs.
- Cross-validation is recommended when both axes are available because it reduces false positives.
- Small images may not contain enough spectral patches for stable statistical testing.
- The selected parameters, especially `patch-size`, `r`, `epsilon`, `jpeg-radius`, and `min-distance`, can influence the result.
- The original image size clues printed by the program are only hints, not exact reconstruction results.

---

## Example Full Command

```bash
python ResamplingDetector.py test.jpg \
  --preprocess tv \
  --axis both \
  --suppress-jpeg \
  --cross-validate \
  --plot-prefix test
```

Expected outputs:

```text
Input image: test.jpg
Image shape: ...
Preprocessing: tv
Patch size: (8, 8)
r: 3
epsilon: 1.0
JPEG suppression: True

=== Detection along axis 0 / vertical ===
...

=== Detection along axis 1 / horizontal ===
...

=== Cross-validation for proportional resampling ===
...

================ FINAL DECISION ================
Final result: RESAMPLED image
...
```
