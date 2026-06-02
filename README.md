
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

## W3 pilots (SUIVI week 3)

Implements the four TODO items: RAISE/fallback PNG subset, Idea 1 (JPEG/x8),
Idea 2 (k ∈ {-1,0,1} + shape metrics), and comparison summary.

```bash
cd 4IM06-G3-Project22

# RAISE-1k: place RAISE_1k.csv in data/raise_raw/ (from unitn download page),
# then download TIFFs and run pilots:
python run_pilots.py --raise-dir data/raise_raw --download --max-images 10

# Or if you already extracted TIFF files into data/raise_raw/tiff/:
python run_pilots.py --raise-dir data/raise_raw --max-images 10

# Or step by step:
python -m pilots.prepare_subset --max-images 5
python -m pilots.idea1_jpeg
python -m pilots.idea2_k_groups
python -m pilots.compare
```

Outputs:

- `data/manifest.csv` — PNG subset metadata
- `data/generated/` — transformed images
- `data/pilot_results/idea1_results.csv`, `idea2_results.csv`
- `data/pilot_results/PILOT_SUMMARY.md` — comparison and recommended direction

Without RAISE TIFFs, `prepare_subset` uses `../img/*.png` and skimage builtins.
