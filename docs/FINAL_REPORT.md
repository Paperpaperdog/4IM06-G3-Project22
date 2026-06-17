# Detecting Image Processing History from Spectral Fingerprints

**Course**: 4IM06 — Image Forensics, Telecom Paris  
**Group**: G3 · Project 22  
**Supervisor**: Quentin Bammey  
**Dataset**: [RAISE-1K](https://loki.disi.unitn.it/RAISE/download.html) (~1,000 RAW camera TIFFs)  
**Date**: June 2026  
**Branch**: `project-integration`

---

## Abstract

We study whether the processing history of a digital image — JPEG compression, geometric downsampling, or upsampling — can be recovered from spectral fingerprints left in the image residual. Our starting point is the a contrario NFA resampling detector of Bammey et al. Reproduction experiments reveal two concrete limitations: the detector is insensitive to pure geometric resampling on real RAISE images (0% significant detections), and even on controlled synthetic data where peaks are detected (91%), the correct source size is recovered as the top-1 candidate in only 11% of cases. These findings motivate a redefined 4-to-6-class task and three complementary methods: (A) a classical DCT+FFT joint detector, (B) a learnable Spectral Mask on a normalised frequency grid, and (C) a Spectral Positional CNN. Mask achieves 56.6% accuracy on the 4-class task (vs. 25% random), CNN achieves 62.5% on the 6-class task. Both methods confirm that ×8 vs. ×16 downsampling constitutes a fundamental Fourier ambiguity: mask overlap = 0.936 across classes, and post-hoc ×8/×16 binary accuracy = 53.9%. CNN, however, achieves F1 = 0.91 for original/JPEG, far above Mask (0.59/0.69), demonstrating that positional encoding and convolution effectively capture JPEG block artifacts. The three routes are complementary: together they systematically quantify the limits of Fourier-only representations and identify where non-Fourier evidence (DCT block energy) is necessary.

---

## 1. Introduction

### 1.1 Motivation

Digital images routinely undergo a chain of operations: geometric resizing, JPEG compression, AI-based enhancement, and re-upload. Image forensics asks whether these operations leave recoverable traces, so that the processing history of an image can be determined from its content alone.

Two operations are particularly relevant:

- **Geometric resampling** (downsampling or upsampling by a factor $f$) introduces periodic correlations in the frequency domain at spatial frequencies related to $f$.
- **JPEG compression** operates on 8×8 DCT blocks, introducing quantisation artifacts that also manifest as peaks in the image spectrum.

The key challenge is that both operations can produce **similar spectral patterns**. Specifically:

| Processing | Typical spectral period (image width $n$) |
|------------|------------------------------------------|
| JPEG 8×8 block coding | Peaks near $n/8,\ n/16,\ \ldots$ |
| ×8 downsampling | Period-8 related structures |
| ×16 downsampling | Period-16 related structures |

When different processing histories produce peaks at the same spectral frequency, the **peak distance $d$ alone is ambiguous** — we cannot recover the processing type from $d$ alone. This is the **Fourier ambiguity** that our project systematically investigates.

### 1.2 Research Questions

1. Can the prior NFA method (Bammey et al.) detect resampling on real camera images?
2. Can Fourier-only representations (log-spectrum) distinguish 4–6 processing classes?
3. What are the specific confusion patterns, and what non-Fourier evidence is needed?

### 1.3 Dataset and Evaluation Protocol

**RAISE-1K**: ~1,000 high-quality camera TIFFs.  
**Train / val / test split**: 700 / 150 / 150 *source images*, preventing crop-level data leakage across splits.  
All deep learning experiments generate per-class patches on-the-fly from the source images within each split.

---

## 2. Prior Method and Reproduction

### 2.1 The NFA Resampling Detector (Bammey et al.)

The reference method operates as follows:

1. Compute the **TV residual** $r = I - \mathrm{TV}(I)$ to suppress smooth content and strong edges.
2. Take the **2D FFT** of $r$ and centre the spectrum.
3. Divide the spectrum into non-overlapping patches; for each candidate distance $d$, compute the **complex Pearson correlation** between a patch and its shifted copy.
4. Count how many patches have a local maximum of correlation at distance $d$ → binomial null model → **NFA (number of false alarms)**.
5. If $\mathrm{NFA}(d) < \varepsilon$, declare distance $d$ significant.

Source size estimation (`candidate_estimation.py`) enumerates candidate original sizes:

$$N = k \cdot C + d \quad \text{or} \quad N = k \cdot C + (C - d)$$

ranked by NFA support.

Our clean-room implementation: `resampling_core.py`.

### 2.2 Reproduction Experiment 1 — W3 Pilot on Real RAISE Images

**Setup**: 10 RAISE images; target size 384×384; 6 conditions per image.

| Condition | Description |
|-----------|-------------|
| `png_identity` | Resize only (identity, PNG) |
| `png_resample_to_target` | Resample from a larger source to 384 |
| `png_sim_x8` | Simulate ×8 grid effect |
| `jpeg_q90_identity` | JPEG Q90, then resize |
| `jpeg_q90_resample_to_target` | JPEG + resample |
| `jpeg_q90_sim_x8` | JPEG + simulated ×8 grid |

**Results**:

| Finding | Evidence |
|---------|----------|
| PNG resampling is indistinguishable | 10/10 images: `png_identity` and `png_resample_to_target` share the **same best peak distance** |
| PNG NFA is not significant | Significant detection rate: **0%**; mean log₁₀(NFA) = −1.19 |
| JPEG + resampling ≈ JPEG alone | `jpeg_q90_identity` and `jpeg_q90_resample_to_target` produce identical NFA curves |
| k-grouping (k ∈ {−1, 0, 1}) is uninformative | Prominence difference: 0.0003 across groups |
| Designed peak (d=128 for 512→384) not dominant | Mean log₁₀(NFA) at d=128: 2.29 (not significant) |

**Interpretation**: The NFA spectral correlation detector is **not sensitive to pure geometric resampling on real RAISE images**. This is not a reproduction failure — it is a finding. Natural image texture in the residual masks the weak periodic resampling signal.

### 2.3 Reproduction Experiment 2 — Controlled Synthetic Dataset (RAISE-100)

To isolate the detector's behaviour from content-related noise, we synthesised a controlled dataset where the ground-truth source size is known.

**Dataset** (`synthesize_controlled_resampling_dataset.py`):
- 100 RAISE images × 3 target sizes (256, 384, 512) × 3 designed peaks × 5 source sizes = **4,500 target images**, all bicubic resampling.

**Results** (N = 9,000 detection rows, vertical + horizontal combined):

| Metric | Value |
|--------|-------|
| Significant peak detected (NFA < 1) | **91.0%** |
| Top-1 source size correct | **11.1%** |
| Top-3 source size correct | 37.5% |
| Best peak distance = designed peak | 23.3% |
| No valid ranking (true rank empty) | 58.5% |

**Interpretation**: On controlled synthetic data the detector frequently finds a significant peak (91%), but **localises the correct source size in only 11% of cases**. The bottleneck is estimation accuracy, not sensitivity. Combined with the W3 result (0% on real images), the prior method faces two distinct failure modes:
- On real images: the signal is too weak to detect.
- On synthetic data: the signal is detected but cannot be localised to the correct source size.

### 2.4 Limitations of the Prior Method — Summary

The two reproduction experiments expose specific limitations:

1. **Real-data insensitivity**: Natural image content overwhelms the weak resampling periodicity, even after TV residual extraction.
2. **Fourier ambiguity**: Even when a peak is detected, multiple processing histories can produce the same (or near-identical) peak distances, making source size recovery unreliable.

These two findings directly motivate the task redefinition and the three routes described below.

---

## 3. Task Redefinition

Based on the reproduction findings, we refine the problem from binary detection ("resampled / not") to **multi-class processing history classification**:

| Class | Generating procedure (for observed size $o$) |
|-------|----------------------------------------------|
| `original` | Crop $o \times o$ directly |
| `JPEG_Q80` | Crop $o \times o$, then JPEG encode at Q=80 |
| `downsample_×8` | Crop $8o \times 8o$, bicubic resize to $o$ |
| `downsample_×16` | Crop $16o \times 16o$, bicubic resize to $o$ |
| `upsample_×4`* | Crop $o/4 \times o/4$, bicubic resize to $o$ |
| `upsample_×8`* | Crop $o/8 \times o/8$, bicubic resize to $o$ |

\* Added in the `n6` unified protocol for all three routes.

The Fourier ambiguity can be decomposed into a hierarchy of difficulties:

| Level | Question | Our finding |
|-------|----------|-------------|
| 1 | Is the image resampled at all? | NFA fails on real data |
| 2 | JPEG vs. downsampling? | CNN: almost never confused (15–19 cases); Mask: moderate |
| 3 | ×8 vs. ×16 downsampling? | **Shared bottleneck** across all methods |
| 4 | Original vs. JPEG? | CNN: F1 ≈ 0.91; Mask: F1 ≈ 0.59/0.69 |

---

## 4. Route A — Classical DCT+FFT Detector

### 4.1 Motivation

The prior NFA method relies on spectral patch correlation, which is sensitive to the global periodicity of the image but not specifically designed to separate JPEG block artifacts from geometric resampling. JPEG compression is characterised by an 8×8 DCT quantisation grid that is **structurally different** from the smooth bicubic resampling kernel.

We therefore design a complementary classical detector (`jpeg_resample_detector.py`) that uses **two independent feature channels** from the image residual:
- **DCT block energy**: measures the strength of block boundary discontinuities characteristic of JPEG.
- **FFT periodicity score**: measures the strength of periodic spectral peaks characteristic of geometric resampling.

### 4.2 Method

**Preprocessing** (`create_forensic_postprocess_dataset.py`): generates 4 categories from raw images:

| Category | Processing |
|----------|-----------|
| `original` | No processing |
| `jpeg` | JPEG encode at Q=85, save as PNG |
| `resample_x8` | 8×8 block-level resampling (inner_delta ±1) |
| `mix` | JPEG then resample, or resample then JPEG |

Note: the ×8 block resampling here is **block-level** (not global bicubic), simulating local grid periodicity.

**Detection pipeline** (`jpeg_resample_detector.py`):

```
Grayscale image → 4-neighbour prediction residual
    ├── DCT block features → JPEG block energy score
    └── FFT periodicity features → resampling period score
         → a contrario NFA (empirical null from clean patches)
              → Label: jpeg_compression / 8x8_resampling / jpeg_and_8x8_resampling / original_or_uncertain
```

**Batch evaluation** (`evaluate_detector_on_dataset.py`): runs the detector over a dataset split, outputs accuracy and confusion matrix. Supports multi-core parallelism (`--workers N`).

### 4.3 Relationship to Prior NFA Method (A0/A1)

| Aspect | Prior NFA (A0/A1) | Route A2 |
|--------|-------------------|-----------|
| Feature | Spectral patch Pearson correlation | DCT block energy + FFT periodicity |
| Goal | Detect resampling period; estimate source size | Classify jpeg / resample_×8 / mix / original |
| Null hypothesis | Binomial NFA on correlation counts | Empirical null distribution from clean images |
| Key difference | Sensitive to **any** periodicity | Separates **block-type** (DCT) from **period-type** (FFT) |

### 4.4 Results

The full pipeline is implemented and integrated. Large-scale quantitative results (accuracy, confusion matrix) are pending execution on a held-out test split. This constitutes one of the two pending items before the final defence.

### 4.5 Size Sweep and Unified Comparison

Route A2 participates in the three-method unified comparison via `unified_method_comparison.py`. Since the detector only addresses period-8 block structures, the comparison uses a **common binary task** (resampled vs. not) at multiple input sizes (32, 64, 96, 128 px), making the three methods strictly comparable on a shared axis.

---

## 5. Route B — Learnable Spectral Mask

### 5.1 Method Design

**Core idea**: patches of different observed sizes $o$ have different native rFFT dimensions. A direct comparison of spectra across sizes would align absolute pixel frequencies rather than physical frequencies. We address this by mapping each log-rFFT spectrum to a **unified normalised frequency grid (512×257, units: cycles/pixel)** so that the same physical frequency aligns to the same grid position regardless of $o$.

Each class $k$ has learnable parameters:
- **Mask** $M_k = \sigma(\text{logits}_k)$: a sigmoid gate over the frequency grid.
- **Reference** $R_k$: a class prototype in frequency space.

Classification score:

$$\text{score}_k = \cos\!\bigl(\mathrm{vec}(x \odot M_k),\ \mathrm{vec}(R_k)\bigr)$$

$$\text{logit}_k = \text{score}_k \cdot e^{s_k} + b_k$$

**Preprocessing pipeline**:

```
RAISE TIFF → random crop (size o) → Y channel → TV residual (weight=0.08)
    → rFFT2 → vertical fftshift → log(1 + |F|)
    → remap to 512×257 normalised frequency grid → DC suppression (σ=3 bins)
    → per-sample normalisation → mask weighting → cosine similarity → 4-class softmax
```

### 5.2 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Classes | 4: original / JPEG_Q80 / downsample_×8 / downsample_×16 |
| Observed sizes $o$ | 128, 96, 64, 48, 32 px |
| Optimizer | AdamW, lr=1e-3 |
| Epochs | 30 (cosine scheduler) |
| Batch size | 64 |
| Train source images | 700 |
| Test samples | **20,000** (4 classes × 5 sizes × 1,000) |

### 5.3 Results

**Overall**:

| Metric | Value |
|--------|-------|
| Test accuracy | **56.6%** (random baseline: 25%) |
| Macro F1 | **0.561** |

**Per-class**:

| Class | F1 | AUC (OvR) |
|-------|----|-----------|
| original | 0.59 | 0.86 |
| JPEG_Q80 | **0.69** | 0.90 |
| downsample_×8 | 0.45 | 0.78 |
| downsample_×16 | 0.51 | 0.82 |

**Accuracy by observed size**:

| 128 px | 96 px | 64 px | 48 px | 32 px |
|--------|-------|-------|-------|-------|
| 63.3% | 60.9% | 56.9% | 54.2% | **47.6%** |

**Confusion matrix (raw counts)**:

```
                pred_orig  pred_JPEG  pred_×8  pred_×16
true_original      2900       1515      278       307
true_JPEG           833       3775      298        94
true_×8             611        365     2122      1902   ← 1902 confused with ×16
true_×16            461        261     1762      2516   ← 1762 confused with ×8
```

**Interpretability — learned mask overlap**:

The pairwise cosine similarity between learned masks across all class pairs has a mean of **0.936**. The model failed to learn class-specific frequency bands. This is a **quantitative signature of Fourier ambiguity**: the log-amplitude spectrum does not contain enough class-discriminative structure at the frequency resolution we use.

### 5.4 Analysis

1. **Fourier-only representations achieve above-chance performance** (56.6% vs. 25%), confirming that some spectral structure is class-specific.
2. **×8 ↔ ×16 is the dominant failure mode**: 1,902 + 1,762 = 3,664 mutual confusions (35–38% of each class), confirming the Fourier ambiguity hypothesis.
3. **original ↔ JPEG confusion is secondary**: 1,515 + 833 bidirectional. The JPEG 8×8 block effect is insufficiently distinct from natural image spectrum in the normalised grid.
4. **JPEG vs. ×8/×16 confusion is limited** (JPEG→×8: 298 cases), suggesting that JPEG and downsampling artifacts are partially separable even in Fourier space.
5. **Smaller patches degrade performance monotonically**: from 63.3% at 128 px to 47.6% at 32 px. This is expected: smaller patches carry fewer spectral coefficients, reducing discriminative power.
6. **High mask overlap (0.936) is a meaningful negative result**: it directly shows that the normalised log-spectrum does not provide class-separable frequency bands, and motivates incorporating non-Fourier evidence (e.g., DCT quantisation tables, phase information).

---

## 6. Route C — Spectral Positional CNN

### 6.1 Design Rationale

Route C investigates whether a deeper model can overcome the Fourier ambiguity encountered by Route B. Key design differences from Mask:

| Dimension | Spectral Mask | Spectral CNN |
|-----------|--------------|--------------|
| Observed size | Multiple $o \in \{32,48,64,96,128\}$ | Fixed 64×64 |
| Spectrum representation | Normalised 512×257 grid | **Native** 64×33 rFFT |
| Positional information | Implicit in grid mapping | **Explicit**: 43-channel sinusoidal encoding |
| Classifier | Per-class mask + cosine similarity | Lightweight convolutional network |
| Completed classes | 4 | **6** (includes ×2, ×4) |

**Positional encoding** (λ ∈ {1, 2, 4, 8, 16, 32}): sinusoidal features encoding the normalised frequency coordinates $(U, V, r, \cos\theta, \sin\theta)$ and their harmonic expansions. This gives the CNN explicit access to **where** in frequency space each coefficient appears — critical for distinguishing period-8 from period-16 structures.

Total input channels: **1** (log-spectrum) **+ 43** (positional encoding) = **44**.

### 6.2 Architecture

```
Input 44×64×33
    → ConvBlock(44→32) → MaxPool
    → ConvBlock(32→64) → MaxPool
    → ConvBlock(64→128) → MaxPool
    → Conv(128→128)
    → Global Average Pooling → Dropout(0.2) → Linear → num_classes
```

Each ConvBlock: 2 × (Conv2d → BatchNorm → GELU).

### 6.3 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Classes | 6: original / JPEG_Q80 / ×2 / ×4 / ×8 / ×16 |
| Observed size | Fixed **64×64** |
| Optimizer | AdamW, lr=3e-4 |
| Epochs | 50 (cosine scheduler) |
| Batch size | 256, AMP enabled |
| Device | GPU (HPC cluster) |
| Test samples | **9,000** (6 classes × 1,500) |

### 6.4 Results

**Overall**:

| Metric | Value |
|--------|-------|
| Test accuracy | **62.5%** |
| Best val accuracy | **65.2%** (≈ epoch 5) |
| Final train accuracy | ~99.1% — **significant overfitting** |

**Per-class performance**:

| Class | F1 | AUC |
|-------|----|-----|
| original | **0.91** | 0.99 |
| JPEG_Q80 | **0.91** | 0.99 |
| downsample_×2 | 0.73 | 0.95 |
| downsample_×4 | 0.41 | 0.77 |
| downsample_×8 | **0.23** | 0.74 |
| downsample_×16 | 0.48 | 0.83 |

**Key confusion pairs** (per-class N = 1,500):

| Confusion pair | Count | Severity |
|----------------|-------|----------|
| JPEG → ×8 | 15 | Minimal |
| JPEG → ×16 | 19 | Minimal |
| ×8 → ×16 | **680** | Severe (recall ×8 = 18.3%) |
| ×16 → ×8 | 301 | High |
| ×4 ↔ ×8 | 202 + 351 | High (adjacent-factor confusion) |
| original ↔ JPEG | 41 + 65 | Controlled |

### 6.5 Post-hoc 4-Class Ablation

To estimate how much the presence of intermediate classes (×2, ×4) affects ×8/×16 discrimination, we extract the 4-class sub-matrix (original / JPEG / ×8 / ×16) from the 6-class confusion matrix **without retraining**:

| Metric | 6-class (full) | 4-class subset (post-hoc) |
|--------|---------------|--------------------------|
| Test accuracy | 62.5% | **76.1%** (inflated: ×2/×4 confusion excluded) |
| Macro F1 | — | **0.713** |
| ×8/×16 binary accuracy | — | **53.9%** (near random) |
| ×8 → ×4 bridge confusion | — | 351 cases |

The 76.1% is inflated because ×2/×4 samples that absorb ×8 predictions are removed from the denominator. The key finding is that **even in the 4-class extraction, ×8/×16 binary accuracy is only 53.9%** — confirming that this confusion is intrinsic to the frequency representation, not an artefact of the 6-class setup.

### 6.6 Analysis

1. **CNN dramatically outperforms Mask on original/JPEG** (F1 = 0.91 vs. 0.59/0.69). TV residual + positional encoding + convolution captures the JPEG 8×8 block artifact effectively.
2. **JPEG vs. ×8/×16 is almost never confused** in CNN (15–19 cases). This contrasts with Mask (298 cases), suggesting that the native spectral representation at 64×33 — with explicit positional encoding — encodes JPEG artifacts in a way that is more separable.
3. **Downsampling factor discrimination remains the shared bottleneck**: ×8 recall = 18.3%, pulled toward ×16, while ×4 acts as an intermediate attractor.
4. **Overfitting**: train accuracy reaches 99% while val accuracy plateaus at 65% after epoch 5. Reports should be based on the best checkpoint.

---

## 7. Cross-Route Comparison

### 7.1 Summary Table

| Dimension | Route A — Classical | Route B — Spectral Mask | Route C — CNN |
|-----------|---------------------|------------------------|---------------|
| Learns parameters? | No | Yes (mask + reference) | Yes (full CNN) |
| Task | JPEG vs. ×8 (binary) | 4 classes, 5 sizes | 6 classes, fixed 64×64 |
| Test samples | Pending | 20,000 | 9,000 |
| Overall accuracy | Pending | **56.6%** | **62.5%** |
| original / JPEG F1 | — | 0.59 / 0.69 | **0.91 / 0.91** |
| ×8 / ×16 F1 | Ambiguous | 0.45 / 0.51 | 0.23 / 0.48 |
| ×8 ↔ ×16 confusion | Spectrally ambiguous | **3,664** / 10,000 | 981 / 3,000 |
| JPEG → ×8 | — | 298 | 15 |
| Key insight | Peak detected but not localised | Fourier-only ceiling quantified | CNN breaks JPEG barrier |

### 7.2 The Three Routes Are Complementary

The routes are not competing alternatives; they illuminate different aspects of the same scientific question:

- **Route A** introduces non-Fourier evidence (DCT block energy channel independent of FFT periodicity), testing whether explicit feature engineering can resolve JPEG vs. resampling without learning.
- **Route B** provides the cleanest test of the Fourier-only hypothesis: with a fixed normalised frequency grid and a linear (cosine similarity) classifier, it measures exactly how much discriminative information the log-amplitude spectrum contains. The answer is: enough to beat chance, but not enough to separate ×8 from ×16 (mask overlap = 0.936).
- **Route C** tests whether a non-linear classifier with explicit frequency position information can recover more. For original/JPEG it can; for ×8/×16, the CNN's native 64×33 spectrum at 64×64 patch size yields a binary accuracy of 53.9% — confirming that the ambiguity is representation-level, not classifier-level.

### 7.3 Why ×8 vs. ×16 Is a Shared Bottleneck

The theoretical explanation is as follows. When a $16o \times 16o$ patch is downsampled to $o \times o$, the spectral envelope is compressed by a factor of 16. When an $8o \times 8o$ patch is downsampled to $o \times o$, the compression factor is 8. In both cases, the resulting $o \times o$ spectrum shows a broadened spectral envelope, but the difference between factor-8 and factor-16 compression is subtle when $o$ is small (32–64 px): the spectral energy distribution is nearly identical. Only at larger $o$ (128+ px) does the difference become visible — consistent with the Mask accuracy curve (63.3% at 128 px vs. 47.6% at 32 px).

Non-Fourier evidence — particularly the **absence or presence of JPEG DCT quantisation artifacts** and the **sharpness of interpolation kernel side-lobes** — is theoretically capable of breaking this ambiguity when combined with the spectral signal.

### 7.4 The Value of Negative Results

Several findings are framed as "failures" but are scientifically informative:

| Negative result | Scientific meaning |
|----------------|-------------------|
| NFA: 0% detection on real RAISE images | Spectral patch correlation is insufficient for natural images without stronger residual extraction |
| NFA: top-1 size accuracy 11% on synthetic | The spectral peak distance $d$ is ambiguous; multiple source sizes are equally consistent |
| Mask overlap = 0.936 | The log-amplitude spectrum alone does not provide class-specific frequency bands — Fourier-only has a hard ceiling |
| CNN ×8 recall = 18.3% | Downsampling factor discrimination needs spectral resolution beyond 64×33, or non-Fourier evidence |

---

## 8. Conclusions and Outlook

### 8.1 What We Found

**① Fourier ambiguity is real and precisely quantified.**  
Spectral Mask overlap across classes = 0.936; ×8 ↔ ×16 mutual confusion = 3,664 / 10,000 cases in Mask; post-hoc CNN binary accuracy = 53.9%. No log-amplitude spectrum method reliably separates ×8 from ×16 downsampling.

**② CNN achieves a breakthrough on original vs. JPEG.**  
SpectralPositionalCNN: F1 = 0.91 for both classes, far above Spectral Mask (F1 = 0.59 / 0.69). TV residual + sinusoidal positional encoding + convolution captures JPEG 8×8 quantisation artifacts in a way that is class-discriminative and robust (JPEG → ×8: only 15 cases).

**③ NFA limitations are precisely characterised.**  
On real RAISE images: 0% significant detections for pure geometric resampling. On controlled synthetic data: 91% peak detection but only 11% top-1 source size accuracy. The bottleneck is localisation (Fourier ambiguity), not sensitivity.

**④ The DCT+FFT pipeline (Route A2) provides complementary non-Fourier evidence.**  
By treating DCT block energy and FFT periodicity as independent feature channels, Route A2 can in principle separate JPEG from geometric resampling without relying on spectral peak distance alone. Quantitative evaluation is pending.

### 8.2 Outlook

| Priority | Work | Status |
|----------|------|--------|
| High | **Route A2 quantitative evaluation**: run `evaluate_detector_on_dataset.py` on held-out test split; obtain accuracy + confusion matrix | Pending |
| High | **4-class CNN retrain** (`v1_final64_poscnn4`): original / JPEG / ×8 / ×16, enabling a fair comparison with Spectral Mask on identical task and split | Pending |
| Medium | **Hybrid Fourier + DCT model**: combine log-spectrum with DCT quantisation block evidence to break the ×8/×16 ambiguity | Not started |
| Low | **Multi-size CNN** (`n6` protocol): train CNN at multiple observed sizes (32–128 px) to match the Mask size-sweep experiment | Configured |

---

## References

1. Bammey et al., *A Non-Parametric Approach to Explain and Predict Image Resampling* — https://bammey.com/resampling_detection.pdf  
2. RAISE-1K Dataset — https://loki.disi.unitn.it/RAISE/download.html  
3. Group weekly logs — [`SUIVI.md`](../SUIVI.md)  
4. W3 pilot analysis — [`REPORT.zh.md`](../REPORT.zh.md)  
5. Experiment data index — [`EXPERIMENT_SUMMARY.md`](../EXPERIMENT_SUMMARY.md)

---

*This report integrates work from branches `main`, `zzy_raise100_resized_dataset`, `test`, `xby-branch`, consolidated under `project-integration`.*
