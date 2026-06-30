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

If you do not have `uv`, create a Python virtual environment and install the project dependencies with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
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
  --model-out artifacts/activity_classifier.joblib
```

## Evaluate

Specify the target activity with the flag `--target`, this activity will be considered `correct` activity

```bash
.venv/bin/python -m src.evaluate \
  --data-dir "datasets/UCI HAR Dataset" \
  --model-path artifacts/activity_classifier.joblib \
  --target WALKING
```

Valid targets are:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`

The evaluator prints:

- six-class activity accuracy
- six-class classification report
- status accuracy for the selected target
- status classification report
- status confusion matrix

## Notes

Run commands from the repository root. The source files use package imports, so prefer `python -m src.train` and `python -m src.evaluate` over running the files directly.
