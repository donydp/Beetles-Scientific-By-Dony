# Sentinel Beetles – HDR ML Challenge (NEON) Solution Repo

This repository packages my **training code**, **final submission bundle**, and **repro steps** for the
**Beetles as Sentinel Taxa: Predicting drought conditions from NEON specimen imagery** challenge.

It was created to satisfy the *Participation Requirements* request from the organizers
(“training code, submission, etc.”).

## Repository Layout

- `training/`
  - Training script(s) used to produce a Codabench submission bundle.
- `submission_bundle/`
  - The exact files required by Codabench:
    - `model.py`
    - `model_weights.pth`
    - `encoders.pkl`
    - `lnt_calibration.npz`
    - `requirements.txt`
    - `requirements.txt.txt`
  - `submission_original.zip` – the uploaded bundle as a single zip.
  - `requirements_codabench_safe.txt` – **empty** template (some Codabench tracks restrict packages).
- `scripts/`
  - `verify_local_inference.py` – small smoke-test that loads `model.py` and runs `predict()`.

## Quick Start (local)

> The official dataset is hosted on Hugging Face: `imageomics/sentinel-beetles`.
> You may need a HF token depending on access rules.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_train.txt
python scripts/verify_local_inference.py
```

## Reproducing a Submission Bundle

The training script is located in `training/` and outputs:
- `artifacts/` (weights + encoders + calibration + model.py)
- `artifacts/submission.zip` (upload this to Codabench)

See the top docstring of the training script for configuration and notes.

## Notes about Codabench dependencies

Some Codabench environments restrict certain packages (e.g., `pillow`) in `requirements.txt`.
If you run into that restriction, use an **empty** requirements file (see
`submission_bundle/requirements_codabench_safe.txt`) and ensure the runtime already provides
image handling, or modify `model.py` to avoid extra dependencies.

## Contact
If organizers need anything else (commit hash, exact command line, environment details), open an issue or contact me.

Generated: 2026-02-27
