# E2: 6-class CNN ablation (from existing metrics)

Source: `CNN/spectral-history-cnn/outputs/v1_final64_poscnn/metrics_test.json`

## Summary

| Metric | Value |
|--------|-------|
| 6-class accuracy | 62.5% |
| 4-class subset accuracy | 76.1% |
| 4-class macro F1 | 0.713 |
| ×8/×16 binary accuracy (true ∈ {×8,×16}) | 53.9% |
| ×8→×16 (6-class) | 680 |
| ×16→×8 (6-class) | 301 |
| ×8→×4 (bridge) | 351 |
| ×4→×8 (bridge) | 202 |

## 4-class subset F1

| Class | F1 | Recall | Support |
|-------|-----|--------|---------|
| original | 0.937 | 94.6% | 1473 |
| JPEG | 0.944 | 93.1% | 1435 |
| downsample_x8 | 0.347 | 27.9% | 987 |
| downsample_x16 | 0.622 | 72.6% | 1204 |

## 4-class confusion matrix

```
[[1394, 41, 5, 33], [65, 1336, 15, 19], [21, 11, 275, 680], [21, 8, 301, 874]]
```

## Key pairs (from metrics JSON)

- JPEG_as_downsample_x16: 19
- JPEG_as_downsample_x8: 15
- JPEG_as_original: 65
- downsample_x16_as_JPEG: 8
- downsample_x16_as_downsample_x8: 301
- downsample_x2_as_original: 15
- downsample_x4_as_downsample_x8: 202
- downsample_x8_as_JPEG: 11
- downsample_x8_as_downsample_x16: 680
- downsample_x8_as_downsample_x4: 351
- original_as_JPEG: 41
- original_as_downsample_x2: 11
