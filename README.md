# Body Movements Quality Assessment

This project trains and evaluates a simple activity classifier for the UCI Human Activity Recognition dataset. It predicts the six original activity classes, then maps predictions into a target-conditioned movement quality status:

- `correct`: the predicted activity matches the selected target movement
- `compensatory`: the predicted activity is another active movement
- `rest`: the predicted activity is a resting posture

## Dataset

Download the Human Activity Recognition Using Smartphones dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extract it into the `datasets` directory so the project has this structure:

```text
datasets/
└── UCI HAR Dataset/
    ├── train/
    │   ├── X_train.txt
    │   ├── y_train.txt
    │   ├── subject_train.txt
    │   └── Inertial Signals/
    └── test/
        ├── X_test.txt
        ├── y_test.txt
        ├── subject_test.txt
        └── Inertial Signals/
```

## Setup

This project uses Python `>=3.13`.

If you use `uv`, install dependencies with:

```bash
uv sync
```

To include the optional CNN dependency with `uv`, run:

```bash
uv sync --extra cnn
```

If you do not have `uv`, create a Python virtual environment and install the project dependencies with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To train the optional 1D CNN on raw inertial time series, install the CNN extra:

```bash
python -m pip install -e ".[cnn]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Train

Run training from the repository root:

```bash
.venv/bin/python -m src.train \
  --data-root "datasets/UCI HAR Dataset" \
  --model-dir artifacts
```

Training first fits and compares the feature-based models on the validation split:

- `logistic_regression`
- `random_forest`
- `linear_svm`

If PyTorch is installed, training also fits `inertial_1d_cnn` on the nine raw `Inertial Signals` channels with input shape `(samples, 9, 128)`. If PyTorch is not installed, the CNN step is skipped and the feature-based models still run.

After validation comparison, each model is refit on `train + validation` before it is saved. The command prints validation accuracy, saves the final trained models, and prints test reports.

Saved model files:

```text
artifacts/
├── logistic_regression.joblib
├── random_forest.joblib
├── linear_svm.joblib
└── inertial_1d_cnn.joblib
```

## Evaluate

Specify the target activity with the flag `--target`, this activity will be considered `correct` activity

```bash
.venv/bin/python -m src.evaluate \
  --data-dir "datasets/UCI HAR Dataset" \
  --model-path artifacts/logistic_regression.joblib \
  --latency-batch-size 1 \
  --latency-repeats 50 \
  --target WALKING
```

For the CNN model, use inertial input:

```bash
.venv/bin/python -m src.evaluate \
  --data-dir "datasets/UCI HAR Dataset" \
  --model-path artifacts/inertial_1d_cnn.joblib \
  --input-kind inertial \
  --latency-batch-size 1 \
  --latency-repeats 50 \
  --target WALKING
```

Valid targets are:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`

The evaluator prints:

- pipeline latency summary with mean, median, P95, and per-sample latency
- six-class activity accuracy
- six-class classification report
- status accuracy for the selected target
- status classification report
- status confusion matrix

Latency options:

- `--latency-batch-size`: number of samples per timed prediction call
- `--latency-warmup`: number of untimed warmup prediction calls
- `--latency-repeats`: number of timed prediction calls
- `--skip-latency`: disable latency measurement

Latency is measured for the full inference pipeline: model prediction plus target-conditioned status mapping.

## Notes

Run commands from the repository root. The source files use package imports, so prefer `python -m src.train` and `python -m src.evaluate` over running the files directly.
