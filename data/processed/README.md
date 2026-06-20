# Processed spectrum cache (not in git)

Mask and CNN share precomputed rFFT spectra under:

```text
data/processed/n6_spectra_size{32,64,96,128}/
```

Generate locally (CPU, ~hours depending on hardware):

```bash
cd 4IM06-G3-Project22
bash scripts/prepare_n6_spectra.sh
```

Or run the full n6 pipeline scripts — they call prepare when the cache is missing.
See [`docs/EXPERIMENT_RUNBOOK.md`](../docs/EXPERIMENT_RUNBOOK.md) §2.
