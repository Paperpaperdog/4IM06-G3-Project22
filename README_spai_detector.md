# SPAI-style Image Post-processing Detector

This project implements a SPAI-inspired image post-processing detector.

It is designed to distinguish original images from images affected by JPEG compression, resampling, and combinations of JPEG compression and resampling.

The detector does not directly analyze image semantics. Instead, it extracts forensic and spectral features from image patches, including FFT-based features, residual-domain features, JPEG block features, DCT coefficient features, and resampling periodicity features.

---

## 1. What This Code Does

The script supports two modes:

```bash
--mode train
--mode predict
```

In training mode, the program starts from a folder of real images and automatically generates five types of image samples:

```text
original
jpeg
resampled
jpeg_then_resampled
resampled_then_jpeg
```

The meaning of each class is:

| Class | Meaning |
|---|---|
| `original` | Original image patch without artificial post-processing |
| `jpeg` | Image patch after simulated JPEG compression |
| `resampled` | Image patch after simulated resizing/resampling |
| `jpeg_then_resampled` | JPEG compression first, then resampling |
| `resampled_then_jpeg` | Resampling first, then JPEG compression |

During training, the code uses random crop to extract fixed-size patches from images. This increases data diversity while avoiding artificial resizing in the preprocessing stage.

During prediction, the code uses center crop to make the output stable and reproducible.

---

## 2. Main Features Extracted

The detector extracts several groups of handcrafted forensic features.

### 2.1 FFT Radial Profile Features

The image is converted to grayscale, and a 2D Fourier transform is applied.

The code computes the log-magnitude spectrum and summarizes its radial energy distribution.

These features describe the global frequency distribution of the image.

They are useful because JPEG compression and resampling may change the natural frequency statistics of an image.

### 2.2 Residual FFT Features

The code first applies a Laplacian high-pass filter to obtain a residual image.

Then it extracts FFT radial profile features from this residual image.

This helps emphasize high-frequency artifacts such as compression noise, block discontinuities, and interpolation traces.

### 2.3 2D FFT Directional Peak Features

In addition to radial FFT statistics, the code also extracts directional spectral peak features from:

```text
horizontal frequency profile
vertical frequency profile
main diagonal profile
anti-diagonal profile
```

This is useful because JPEG block artifacts and resampling artifacts often create directional or periodic structures in the frequency domain.

### 2.4 JPEG 8×8 Block Boundary Features

JPEG compression operates on 8×8 blocks.

The code compares pixel differences on 8×8 block boundaries with non-boundary regions.

If the boundary discontinuity is strong, the image may contain JPEG-like block artifacts.

### 2.5 DCT Coefficient Features

Since JPEG is based on 8×8 block DCT transform and quantization, the code also extracts DCT-domain statistics.

The extracted features include:

```text
mean absolute DCT coefficients
DC coefficient statistics
low / mid / high frequency energy
high-frequency energy ratios
small AC coefficient ratios
```

These features help detect JPEG compression traces, especially high-frequency suppression and DCT quantization effects.

### 2.6 Resampling Periodicity Features

The code computes the Laplacian residual and then calculates 1D autocorrelation along horizontal and vertical directions.

Resampling often introduces periodic correlations due to interpolation.

The code summarizes autocorrelation peaks to capture these periodic structures.

---

## 3. Installation

Install the required Python packages:

```bash
pip install opencv-python numpy scipy scikit-learn joblib tqdm
```

Required libraries:

```text
opencv-python
numpy
scipy
scikit-learn
joblib
tqdm
```

---

## 4. Dataset Preparation

The training data folder should contain real images.

Example folder structure:

```text
data/
└── split_dataset/
    ├── train/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    └── test/
        ├── 000801.png
        ├── 000802.png
        └── ...
```

The training folder should contain original real images.

The script will automatically generate JPEG, resampled, JPEG-then-resampled, and resampled-then-JPEG variants during training.

---

## 5. How to Train

Example command:

```bash
python spai_detector_new.py 
  --mode train 
  --data_dir data/split_dataset/train 
  --model_path spai_style_detector.pkl 
  --image_size 256 
  --variants_per_image 2
```


### Parameter Explanation

| Parameter | Meaning |
|---|---|
| `--mode train` | Run the script in training mode |
| `--data_dir` | Folder containing original training images |
| `--model_path` | Path to save the trained model |
| `--image_size` | Fixed crop size used for feature extraction |
| `--variants_per_image` | Number of generated variants per class for each image |
| `--max_images` | Optional limit for quick debugging |

For example:

```bash
--variants_per_image 2
```

means that for each real image, the code generates two samples for each class:

```text
2 original patches
2 jpeg patches
2 resampled patches
2 jpeg_then_resampled patches
2 resampled_then_jpeg patches
```

Since there are 5 classes, one image contributes:

```text
5 × 2 = 10 training samples
```

---

## 6. How to Predict

After training, use the saved `.pkl` model to predict images in a test folder.

```bash
python spai_detector_new.py 
  --mode predict 
  --data_dir data/split_dataset/test 
  --model_path spai_style_detector.pkl 
  --output_csv predictions.csv
```

The prediction results will be saved into:

```text
predictions.csv
```

---

## 7. Training Output

During training, the program prints:

```text
Feature matrix: (N, D)
Labels: [...]
Classification Report
Confusion Matrix
Saved model to: spai_style_detector.pkl
```

### Feature Matrix

Example:

```text
Feature matrix: (10000, 298)
```

This means:

```text
10000 training samples
298 extracted features per sample
```

The exact feature dimension may depend on the implemented feature groups.

### Classification Report

The classification report includes:

```text
precision
recall
f1-score
support
```

Interpretation:

| Metric | Meaning |
|---|---|
| `precision` | Among samples predicted as this class, how many are correct |
| `recall` | Among true samples of this class, how many are correctly found |
| `f1-score` | Harmonic mean of precision and recall |
| `support` | Number of validation samples in this class |

A higher F1-score means better classification performance.

### Confusion Matrix

The confusion matrix follows this class order:

```text
original
jpeg
resampled
jpeg_then_resampled
resampled_then_jpeg
```

Rows represent true labels.

Columns represent predicted labels.

For example:

```text
[[180  10   5   2   3]
 [ 15 160  10   5  10]
 [ 20  10 150  15   5]
 [ 10  12  20 130  28]
 [  8  20  12  25 135]]
```

The diagonal values are correct predictions.

Off-diagonal values are misclassifications.

---

## 8. Prediction CSV Format

The output CSV contains one row per image.

Columns include:

```text
path
prediction
prob_original
prob_jpeg
prob_resampled
prob_jpeg_then_resampled
prob_resampled_then_jpeg
```

Example:

```text
path,prediction,prob_original,prob_jpeg,prob_resampled,prob_jpeg_then_resampled,prob_resampled_then_jpeg
data/test/000001.png,original,0.72,0.10,0.08,0.04,0.06
```

### Column Meaning

| Column | Meaning |
|---|---|
| `path` | Image file path |
| `prediction` | Final predicted class |
| `prob_original` | Probability of original |
| `prob_jpeg` | Probability of JPEG compression |
| `prob_resampled` | Probability of resampling |
| `prob_jpeg_then_resampled` | Probability of JPEG followed by resampling |
| `prob_resampled_then_jpeg` | Probability of resampling followed by JPEG |
