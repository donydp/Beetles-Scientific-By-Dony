# Sentinel Beetles — Event-level SPEI Prediction (HDR ML Challenge 2025)

This repository contains training code and a Codabench-compatible submission bundle for the **“Beetles as Sentinel Taxa: Predicting drought conditions from NEON specimen imagery”** challenge (Imageomics Institute + NEON).

The model predicts three drought metrics for each sampling event:
- `SPEI_30d`
- `SPEI_1y`
- `SPEI_2y`

Each prediction includes uncertainty (Gaussian mean `mu` and standard deviation `sigma`) for CRPS evaluation.

---

## Approach (High-level)

- **Event-level learning**: multiple specimen images (up to 4) are pooled to make **one** prediction per event.
- **Backbone**: `MobileNetV3-Small` (fast inference).
- **Pooling**: per-event **mean + max** pooling over specimen embeddings.
- **Metadata embeddings**: embeddings for `scientificName`, `domainID`, `siteID`.
- **Priors** (optional but enabled): learned priors computed from training statistics for species/domain/site (standardized).
- **Saab folding**: compute a PCA/saab-like linear projection from backbone features and **fold** it into a single `Linear` layer (`saab_linear`) to keep inference simple.
- **Optimization**: `SAM` (Sharpness-Aware Minimization) for robustness. AMP autocast enabled on CUDA; `GradScaler` disabled when SAM is enabled to avoid unscale-stage issues.
- **Uncertainty calibration**:
  - affine calibration for mean (`A`, `b`)
  - sigma scaling from residuals
  - optional variance temperature scaling (`var_temp`)
  - long-horizon sigma floor for `SPEI_1y` and `SPEI_2y` to reduce overconfidence under domain shift

---

## Repository Contents

- `ziptrue_final.py` — training script that also packages Codabench submission
- `artifacts_pack/` — created after training
  - `model.py`
  - `model_weights.pth`
  - `encoders.pkl`
  - `lnt_calibration.npz`
  - `requirements.txt` (empty)
  - `requirements.txt.txt` (empty)
- `submission.zip` — Codabench submission bundle created by the script

---

## Environment Setup

### Recommended: venv
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### Install dependencies (training)
```bash
pip install torch torchvision datasets tqdm pillow numpy
```

> Notes:
> - `pillow` is required **locally** for training (image decoding), but on Codabench the runtime may restrict installing packages.
> - This is why the submission bundle includes **empty** `requirements.txt` files, relying on the platform’s preinstalled runtime dependencies.

---

## Training + Submission Packaging

Run the script:

```bash
python ziptrue_final.py
```

The script will:
1. download/load Hugging Face dataset: `imageomics/sentinel-beetles`
2. build event-level indices
3. split into:
   - **train**
   - **ID validation** (within-domain split)
   - **OOD validation** (HF validation split if available; otherwise fallback)
4. fit the folded Saab/PCA projection
5. train with SAM + EMA
6. calibrate uncertainty on ID-val (affine + sigma_scale + optional var_temp)
7. write `model.py` for Codabench inference
8. create `submission.zip`

When finished, you should see:
- `submission.zip` in repo root
- and artifacts in `artifacts_pack/`

---

## Output Format (Codabench)

The `model.py` inside the submission bundle exposes:

- `Model.load()` — loads weights + encoders + calibration
- `Model.predict(event_records)` — returns:

```json
{
  "SPEI_30d": {"mu": 1.2, "sigma": 0.1},
  "SPEI_1y":  {"mu": -0.3, "sigma": 0.7},
  "SPEI_2y":  {"mu": 2.2, "sigma": 0.3}
}
```

`predict()` takes an **event** (list of dicts). Each dict contains:
- `relative_img` (PIL.Image)
- `colorpicker_img` (PIL.Image)
- `scalebar_img` (PIL.Image)
- `scientificName` (str)
- `domainID` (int)
- (optionally) `siteID` (may be absent in test data)

The inference code handles missing fields by falling back to defaults.

---

## Key Configuration Values (in `ziptrue_final.py`)

- `IMAGE_SIDE = 160`
- `MAX_IMAGES_PER_EVENT = 4`
- `BATCH_SIZE = 16`
- `EPOCHS = 60`
- `USE_SAM = True` (`SAM_RHO = 0.05`)
- `USE_AMP = True` (autocast on CUDA)
- `USE_EMA = True` (`EMA_DECAY = 0.999`)
- Saab folding:
  - `SAAB_K_DIM = 192`
  - `SAAB_ENERGY_THRESHOLD = 0.997`
- Calibration:
  - `USE_VARIANCE_TEMP = True`
  - `SIGMA_FLOOR_LONG_STD_FRAC = 0.06`

---

## Reproducibility Notes

- Seed is fixed (`SEED = 42`) for python/numpy/torch.
- `torch.backends.cudnn.benchmark = True` for speed; exact determinism may vary across GPUs/drivers.
- Hugging Face dataset access may require authentication depending on the dataset settings.

---

## Troubleshooting

### 1) Hugging Face authentication / dataset access
If the dataset requires access:
- set `HF_TOKEN` in the script, or export it:
```bash
export HF_TOKEN=YOUR_TOKEN
```

### 2) CUDA / AMP issues
- AMP is used only when `cuda` is available.
- When `USE_SAM=True`, `GradScaler` is disabled by design to avoid optimizer-stage errors.

### 3) “pillow is not an allowed package” on Codabench
Some Codabench environments restrict installing packages via `requirements.txt`.
This submission bundle uses empty `requirements.txt` to avoid pip installs.

If Codabench still complains, verify your platform’s base environment already includes PIL / Pillow. If not, contact organizers to whitelist pillow.

### 4) Predictions file missing in scoring logs
Codabench uses your `model.py` + harness to generate `predictions.csv`.
If you see:
`cannot open file '/app/input//res/predictions.csv'`
it usually means inference crashed earlier. Check prediction logs for the real exception.

---

## License

This repository is provided for challenge participation and reproducibility. Dataset licensing follows the challenge dataset’s terms (CC BY 4.0 as stated by the organizers).

---

## Acknowledgements

- Imageomics Institute + NEON for dataset and challenge design.
- NEON DP1.10022.001 beetle specimen imagery and GRIDMET drought products for SPEI targets.
