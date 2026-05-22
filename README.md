
Minimal clean-room implementation of the core method in
`resampling_detection (1).pdf`.

## Files

- `resampling_core.py`: core implementation:
  - TV residual extraction;
  - Fourier spectrum computation;
  - complex spectral patch correlation;
  - a-contrario NFA computation.
- `demo_resampling_detection.py`: small runnable demo.

## Run

```bash
python3 demo_resampling_detection.py
```

With no argument, the demo resamples `skimage.data.camera` from `512 x 512`
to `384 x 384`. The expected anomalous distances are:

- `512 mod 384 = 128`
- `-512 mod 384 = 256`

You can also run it on your own image:

```bash
python3 demo_resampling_detection.py path/to/image.png
# or: use default test image which is skimage.data.camera()
python3 demo_resampling_detection.py
```

## Method Summary

1. TV denoising estimates the smooth/edge image component.
2. The residual is `image - TV(image)`, which keeps noise-like traces and small
   resampling oscillations.
3. The residual is transformed with a 2D FFT.
4. The spectrum is split into non-overlapping patches.
5. For each distance `d`, every patch is compared with its circularly shifted
   counterpart using complex Pearson correlation.
6. For each `d`, the method counts how many patches have a local maximum
   correlation inside `[d-r, d+r]`.
7. Under the null hypothesis of no resampling, this count follows a binomial
   law with probability `1 / (2r + 1)`.
8. The NFA is the binomial tail probability multiplied by the number of tested
   distances. Small NFA means an unusually repeated spectral correlation.
