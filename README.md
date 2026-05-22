# 4IM06-G3-Project22

## Resampling Detector CLI

This project now includes `detect_resampling.py`, a command-line tool that:
- takes an input image,
- outputs a `RESAMPLED` / `NOT_RESAMPLED` decision,
- saves an NFA plot (`.png`).

### Run

```powershell
python detect_resampling.py --image path\to\your\image.png
```

### Useful options

```powershell
python detect_resampling.py `
  --image path\to\your\image.png `
  --outdir outputs `
  --residual rank `
  --rank-window 7 `
  --patch-h 16 `
  --patch-w 16 `
  --nfa-threshold 1e-3
```

### Output

- Console prints:
  - best vertical NFA
  - best horizontal NFA
  - final decision
- Plot saved to:
  - `outputs/<image_name>_nfa.png`
