import argparse
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report

from .cnn import InertialCnnClassifier, TorchNotInstalledError
from .model import build_activity_classifiers, save_model_to_file
from .data import (
    DatasetSplit,
    load_inertial_test_split,
    load_inertial_train_val_split,
    load_train_val_split,
    load_test_split,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help='Path to the extracted "UCI HAR Dataset" directory.',
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts",
        help="Directory where all trained models will be saved.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def train_model(train_split, model):
    model.fit(train_split.windows, train_split.labels)
    return model


def evaluate_model(model, split, split_name: str):
    predictions = model.predict(split.windows)
    accuracy = accuracy_score(split.labels, predictions)
    print(f"{split_name} accuracy: {accuracy:.4f}")
    print(
        classification_report(
            split.labels,
            predictions,
            digits=4,
            zero_division=0,
        )
    )
    return accuracy


def save_trained_model(model_name: str, model, model_dir: Path) -> Path:
    model_path = model_dir / f"{model_name}.joblib"
    save_model_to_file(model, str(model_path))
    return model_path


def combine_splits(first_split, second_split) -> DatasetSplit:
    return DatasetSplit(
        windows=np.concatenate([first_split.windows, second_split.windows], axis=0),
        labels=np.concatenate([first_split.labels, second_split.labels], axis=0),
        subjects=np.concatenate([first_split.subjects, second_split.subjects], axis=0),
    )


def main():
    args = parse_args()
    train_split, val_split = load_train_val_split(root_dir=args.data_root, random_state=args.random_state)
    test_split = load_test_split(root_dir=args.data_root)
    full_train_split = combine_splits(train_split, val_split)
    model_dir = Path(args.model_dir)

    print(f"Train samples: {train_split.windows.shape}")
    print(f"Validation samples: {val_split.windows.shape}")
    print(f"Final train samples: {full_train_split.windows.shape}")
    print(f"Test samples: {test_split.windows.shape}")

    results = []
    models = build_activity_classifiers(random_seed=args.random_state)
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        train_model(train_split, model)
        val_predictions = model.predict(val_split.windows)
        val_accuracy = accuracy_score(val_split.labels, val_predictions)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        print("Refitting on train + validation before saving...")
        final_model = clone(model)
        train_model(full_train_split, final_model)
        model_path = save_trained_model(model_name, final_model, model_dir)
        results.append((model_name, val_accuracy, final_model, model_path))
        print(f"Saved model to: {model_path}")

    results.sort(key=lambda result: result[1], reverse=True)

    print("\n=== Validation Comparison ===")
    for rank, (model_name, val_accuracy, _, model_path) in enumerate(results, start=1):
        print(f"{rank}. {model_name}: {val_accuracy:.4f} ({model_path})")

    print("\n=== Test Reports ===")
    for model_name, _, model, _ in results:
        print(f"\n--- {model_name} ---")
        evaluate_model(model, test_split, "Test")

    print("\n=== Training inertial_1d_cnn ===")
    inertial_train_split, inertial_val_split = load_inertial_train_val_split(
        root_dir=args.data_root,
        random_state=args.random_state,
    )
    inertial_test_split = load_inertial_test_split(root_dir=args.data_root)
    inertial_full_train_split = combine_splits(inertial_train_split, inertial_val_split)

    print(f"Inertial train samples: {inertial_train_split.windows.shape}")
    print(f"Inertial validation samples: {inertial_val_split.windows.shape}")
    print(f"Inertial final train samples: {inertial_full_train_split.windows.shape}")
    print(f"Inertial test samples: {inertial_test_split.windows.shape}")

    cnn_model = InertialCnnClassifier(
        random_seed=args.random_state,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
    )
    train_model(inertial_train_split, cnn_model)
    cnn_val_predictions = cnn_model.predict(inertial_val_split.windows)
    cnn_val_accuracy = accuracy_score(inertial_val_split.labels, cnn_val_predictions)
    print(f"Validation accuracy: {cnn_val_accuracy:.4f}")

    print("Refitting on train + validation before saving...")
    final_cnn_model = InertialCnnClassifier(
        random_seed=args.random_state,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
    )
    train_model(inertial_full_train_split, final_cnn_model)
    cnn_model_path = save_trained_model("inertial_1d_cnn", final_cnn_model, model_dir)
    print(f"Saved model to: {cnn_model_path}")

    print("\n--- inertial_1d_cnn ---")
    evaluate_model(final_cnn_model, inertial_test_split, "Test")


if __name__ == "__main__":
    main()
