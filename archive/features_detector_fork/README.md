# JPEG Compression and Bicubic ×8 Upsampling Detection

This project provides a simple forensic image-processing pipeline for distinguishing:

1. original images
2. JPEG-compressed images
3. bicubic ×8 upsampled images

The project contains three Python files:

```text
create_dataset.py
jpeg_upsample_detector.py
dataset_evaluator.py
```

---

## 1. Project Structure

```text
project/
│
├── create_dataset.py
├── jpeg_upsample_detector.py
├── dataset_evaluator.py
│
├── input_images/
│   ├── image_001.png
│   ├── image_002.jpg
│   └── ...
│
└── forensic_dataset/
    ├── original/
    ├── jpeg/
    └── upsample_x8/
```

---

## 2. Install Requirements

Before running the code, install the required packages:

```bash
pip install numpy pillow scipy matplotlib tqdm
```

Main dependencies:

```text
numpy
Pillow
scipy
matplotlib
tqdm
```

`tqdm` is only used for progress bars. If it is not installed, the code can still run without progress visualization.

---

## 3. Step 1: Prepare Input Images

Put your clean original images into one folder, for example:

```text
input_images/
    image_001.png
    image_002.jpg
    image_003.bmp
```

Supported image formats include:

```text
.png
.jpg
.jpeg
.bmp
.tif
.tiff
```

---

## 4. Step 2: Generate the Dataset

Use `create_dataset.py` to generate three classes of images:

```text
original/
jpeg/
upsample_x8/
```

Run:

```bash
python create_dataset.py \
    --input_dir input_images \
    --output_dir forensic_dataset \
    --quality 85 \
    --include_original \
    --base_crop_size 256
```

On Windows PowerShell, you can write it in one line:

```bash
python create_dataset.py --input_dir input_images --output_dir forensic_dataset --quality 85 --include_original --base_crop_size 256
```

After running, the output folder will be:

```text
forensic_dataset/
    original/
        image_001_original_crop256_p8.png
        ...
    jpeg/
        image_001_jpeg_q85.png
        ...
    upsample_x8/
        image_001_upsample_x8_bicubic.png
        ...
```

---

## 5. Important Parameters in `create_dataset.py`

### `--input_dir`

Path to the folder containing original input images.

Example:

```bash
--input_dir input_images
```

---

### `--output_dir`

Path to the output dataset folder.

Example:

```bash
--output_dir forensic_dataset
```

---

### `--quality`

JPEG quality factor.

Example:

```bash
--quality 85
```

Lower quality means stronger JPEG compression artifacts.

For example:

```text
quality = 95: weak JPEG artifacts
quality = 85: medium JPEG artifacts
quality = 70: stronger JPEG artifacts
```

---

### `--include_original`

If this option is used, the script will also save the cropped original images.

Example:

```bash
--include_original
```

This is recommended, because the evaluator needs an `original/` folder.

---

### `--base_crop_size`

Top-left crop size before generating JPEG and upsampled versions.

Example:

```bash
--base_crop_size 256
```

This is important because ×8 upsampling creates very large images.

For example:

```text
256 × 256 → 2048 × 2048
512 × 512 → 4096 × 4096
```

Recommended value:

```bash
--base_crop_size 256
```

Use `512` only if your computer has enough memory.

---

## 6. Step 3: Run Detector on One Image

Use `jpeg_upsample_detector.py` to classify a single image.

Example for a JPEG image:

```bash
python jpeg_upsample_detector.py \
    --image forensic_dataset/jpeg/image_001_jpeg_q85.png \
    --null_dir forensic_dataset/original \
    --max_size 1024
```

Windows PowerShell one-line version:

```bash
python jpeg_upsample_detector.py --image forensic_dataset/jpeg/image_001_jpeg_q85.png --null_dir forensic_dataset/original --max_size 1024
```

Example for an upsampled image:

```bash
python jpeg_upsample_detector.py --image forensic_dataset/upsample_x8/image_001_upsample_x8_bicubic.png --null_dir forensic_dataset/original --max_size 1024
```

The output will include:

```text
Final Decision
JPEG score
Resample score
Generic R score
Upsample score
Feature NFAs
```

Possible final labels are:

```text
jpeg_compression
upsample_x8
original_or_uncertain
```

---

## 7. Important Parameters in `jpeg_upsample_detector.py`

### `--image`

The image to be classified.

Example:

```bash
--image forensic_dataset/jpeg/image_001_jpeg_q85.png
```

---

### `--null_dir`

Folder used to build the null distribution.

Usually, use the `original/` folder:

```bash
--null_dir forensic_dataset/original
```

The null distribution is used as a clean-image reference. The detector compares the observed image features with this reference distribution.

---

### `--max_size`

Maximum top-left crop size used by the detector.

Example:

```bash
--max_size 1024
```

The detector does not resize the image. It only crops the image from the top-left corner. This avoids introducing extra interpolation artifacts.

---

### `--theta_jpeg`

Threshold for JPEG detection.

Default:

```bash
--theta_jpeg 3.0
```

Larger value means stricter JPEG detection.

---

### `--theta_resample`

Threshold for upsample detection.

Default:

```bash
--theta_resample 3.0
```

Larger value means stricter upsample detection.

---

### `--delta`

Margin used when both JPEG and upsample scores are high.

Default:

```bash
--delta 2.0
```

If both JPEG and upsample evidence are strong, this parameter controls which class is selected.

---

## 8. Step 4: Evaluate the Whole Dataset

Use `dataset_evaluator.py` to evaluate all images in the dataset.

Run:

```bash
python dataset_evaluator.py \
    --detector jpeg_upsample_detector.py \
    --dataset_root forensic_dataset \
    --null_dir forensic_dataset/original \
    --max_size 1024 \
    --max_null_images 30
```

Windows PowerShell one-line version:

```bash
python dataset_evaluator.py --detector jpeg_upsample_detector.py --dataset_root forensic_dataset --null_dir forensic_dataset/original --max_size 1024 --max_null_images 30
```

The dataset folder must have this structure:

```text
forensic_dataset/
    original/
    jpeg/
    upsample_x8/
```

The evaluator will output:

```text
Evaluation Result
Evaluated: ...
Failures: ...
Correct: ...
Accuracy: ...
Confusion Matrix
```

It will also save two images:

```text
confusion_matrix_heatmap.png
confusion_matrix_heatmap_normalized.png
```

---

## 9. Important Parameters in `dataset_evaluator.py`

### `--detector`

Path to the detector file.

Example:

```bash
--detector jpeg_upsample_detector.py
```

---

### `--dataset_root`

Path to the generated dataset.

Example:

```bash
--dataset_root forensic_dataset
```

The dataset root should contain:

```text
original/
jpeg/
upsample_x8/
```

---

### `--null_dir`

Folder used to build the null distribution.

Recommended:

```bash
--null_dir forensic_dataset/original
```

---

### `--max_per_class`

Maximum number of images evaluated per class.

Example:

```bash
--max_per_class 100
```

This is useful for quick testing.

If you want to evaluate all images, you can omit this parameter.

---

### `--max_null_images`

Number of original images used to build the null distribution.

Example:

```bash
--max_null_images 30
```

A larger value may make the null distribution more stable, but it will also make evaluation slower.

---

### `--print_each`

Print prediction details for every image.

Example:

```bash
--print_each
```

This will show output like:

```text
image_001.png: true=jpeg, pred=jpeg, jpeg=4.2300, resample=1.5200, up=0.9300
```

This is useful for debugging misclassified images.

---

### `--heatmap_path`

Output path for the raw confusion matrix heatmap.

Example:

```bash
--heatmap_path confusion_matrix_heatmap.png
```

---

### `--normalized_heatmap_path`

Output path for the normalized confusion matrix heatmap.

Example:

```bash
--normalized_heatmap_path confusion_matrix_heatmap_normalized.png
```

---

## 10. Full Example Workflow

### Step 1: Generate dataset

```bash
python create_dataset.py --input_dir input_images --output_dir forensic_dataset --quality 85 --include_original --base_crop_size 256
```

### Step 2: Test one JPEG image

```bash
python jpeg_upsample_detector.py --image forensic_dataset/jpeg/image_001_jpeg_q85.png --null_dir forensic_dataset/original --max_size 1024
```

### Step 3: Test one upsampled image

```bash
python jpeg_upsample_detector.py --image forensic_dataset/upsample_x8/image_001_upsample_x8_bicubic.png --null_dir forensic_dataset/original --max_size 1024
```

### Step 4: Evaluate the full dataset

```bash
python dataset_evaluator.py --detector jpeg_upsample_detector.py --dataset_root forensic_dataset --null_dir forensic_dataset/original --max_size 1024 --max_null_images 30
```

---

## 11. How to Read the Results

The evaluator prints a confusion matrix:

```text
true \ pred     original        jpeg            upsample_x8
original        ...
jpeg            ...
upsample_x8     ...
```

Rows are true labels.

Columns are predicted labels.

For example:

```text
jpeg → upsample_x8
```

means that a JPEG image was wrongly classified as an upsampled image.

The normalized heatmap is usually easier to read because each row is converted into percentages or ratios.

---

## 12. Common Problems

### Problem 1: Missing `original/` folder

If you did not use `--include_original` when creating the dataset, the evaluator may not work correctly.

Solution:

```bash
python create_dataset.py --input_dir input_images --output_dir forensic_dataset --quality 85 --include_original --base_crop_size 256
```

---

### Problem 2: Upsampled images are too large

If `--base_crop_size` is too large, ×8 upsampled images may become very large.

Solution:

Use:

```bash
--base_crop_size 256
```

instead of:

```bash
--base_crop_size 512
```

---

### Problem 3: Evaluation is slow

Possible reasons:

- too many images
- large `--max_size`
- too many null images
- very large upsampled images

Solutions:

```bash
--max_per_class 100
--max_size 512
--max_null_images 10
```

Example:

```bash
python dataset_evaluator.py --detector jpeg_upsample_detector.py --dataset_root forensic_dataset --null_dir forensic_dataset/original --max_per_class 100 --max_size 512 --max_null_images 10
```

---

### Problem 4: JPEG and upsample classes are confused

This can happen because JPEG compression and bicubic ×8 upsampling may both create period-8 related traces.

Possible adjustments:

```bash
--theta_jpeg 3.5
--theta_resample 3.5
--delta 2.5
```

Example:

```bash
python dataset_evaluator.py --detector jpeg_upsample_detector.py --dataset_root forensic_dataset --null_dir forensic_dataset/original --theta_jpeg 3.5 --theta_resample 3.5 --delta 2.5
```

If many JPEG images are predicted as `upsample_x8`, try increasing `theta_resample`.

If many upsampled images are predicted as `jpeg`, try increasing `theta_jpeg`.

---

## 13. Recommended Quick Test Commands

For a fast test, use:

```bash
python create_dataset.py --input_dir input_images --output_dir forensic_dataset --quality 85 --include_original --base_crop_size 256
```

Then:

```bash
python dataset_evaluator.py --detector jpeg_upsample_detector.py --dataset_root forensic_dataset --null_dir forensic_dataset/original --max_per_class 50 --max_size 512 --max_null_images 10 --print_each
```

This evaluates only 50 images per class and prints detailed prediction results.

---

## 14. Brief Pipeline Explanation

The pipeline has three stages.

First, `create_dataset.py` creates a controlled dataset. Each original image is top-left cropped. Then the script saves a clean original version, a JPEG-compressed version, and a bicubic ×8 upsampled version.

Second, `jpeg_upsample_detector.py` extracts forensic features from one image. These features describe JPEG block artifacts, DCT coefficient statistics, interpolation smoothness, residual autocorrelation, phase-8 imbalance, and frequency-domain periodic peaks.

Third, `dataset_evaluator.py` applies the detector to the whole dataset. It compares predicted labels with true folder labels and outputs accuracy, a confusion matrix, and heatmap visualizations.

---

## 15. Limitations

This method is an interpretable feature-based detector, not a trained deep learning classifier.

Its performance depends on:

- image content
- JPEG quality
- crop size
- threshold values
- number of null images
- similarity between JPEG artifacts and upsampling artifacts

JPEG compression and bicubic ×8 upsampling can sometimes produce similar period-8 patterns, so misclassification may happen.

Therefore, this method is suitable as an interpretable baseline, but it may not be robust enough for general real-world forensic classification.