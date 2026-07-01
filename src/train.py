import argparse
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report

from .cnn import InertialCnnClassifier, TorchNotInstalledError
from .constants import ACTIVE_ACTIVITIES, ACTIVITY_NAME_TO_ID
from .model import build_activity_classifiers, save_model_to_file
from .data import (
    DatasetSplit,
    load_inertial_test_split,
    load_inertial_train_val_split,
    load_train_val_split,
    load_test_split,
    relabel_split,
)
from .status_mapping import map_many_to_status

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
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument(
        "--label-mode",
        choices=("six_class", "four_class", "both"),
        default="both",
        help="Train with original 6 labels, collapsed 4 labels, or both.",
    )
    parser.add_argument(
        "--target",
        default="WALKING",
        help="Target activity used for comparable validation status accuracy.",
    )
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


def evaluate_status_accuracy(model, split, target_id: int) -> float:
    predictions = model.predict(split.windows)
    true_status = map_many_to_status(split.labels, target_id)
    predicted_status = map_many_to_status(predictions, target_id)
    return accuracy_score(true_status, predicted_status)


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


def selected_label_modes(label_mode: str) -> list[str]:
    if label_mode == "both":
        return ["six_class", "four_class"]
    return [label_mode]


def train_and_save_model(
    model_name: str,
    model,
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    full_train_split: DatasetSplit,
    model_dir: Path,
    target_id: int,
):
    print(f"\n=== Training {model_name} ===")
    train_model(train_split, model)
    val_predictions = model.predict(val_split.windows)
    val_accuracy = accuracy_score(val_split.labels, val_predictions)
    val_status_accuracy = accuracy_score(
        map_many_to_status(val_split.labels, target_id),
        map_many_to_status(val_predictions, target_id),
    )
    print(f"Validation class accuracy: {val_accuracy:.4f}")
    print(f"Validation status accuracy: {val_status_accuracy:.4f}")

    print("Refitting on train + validation before saving...")
    final_model = clone(model)
    train_model(full_train_split, final_model)
    model_path = save_trained_model(model_name, final_model, model_dir)
    print(f"Saved model to: {model_path}")
    return {
        "model_name": model_name,
        "class_accuracy": val_accuracy,
        "status_accuracy": val_status_accuracy,
        "model": final_model,
        "model_path": model_path,
    }


def main():
    args = parse_args()
    target = args.target.strip().upper()
    if target not in ACTIVE_ACTIVITIES:
        raise ValueError(
            f"Invalid target: {args.target}. Must be one of "
            f"{list(ACTIVE_ACTIVITIES)}"
        )
    target_id = ACTIVITY_NAME_TO_ID[target]
    label_modes = selected_label_modes(args.label_mode)

    train_split, val_split = load_train_val_split(root_dir=args.data_root, random_state=args.random_state)
    test_split = load_test_split(root_dir=args.data_root)
    full_train_split = combine_splits(train_split, val_split)
    model_dir = Path(args.model_dir)

    print(f"Train samples: {train_split.windows.shape}")
    print(f"Validation samples: {val_split.windows.shape}")
    print(f"Final train samples: {full_train_split.windows.shape}")
    print(f"Test samples: {test_split.windows.shape}")
    print(f"Label modes: {', '.join(label_modes)}")
    print(f"Status comparison target: {target}")

    results = []
    for label_mode in label_modes:
        mode_train_split = relabel_split(train_split, label_mode)
        mode_val_split = relabel_split(val_split, label_mode)
        mode_full_train_split = relabel_split(full_train_split, label_mode)
        models = build_activity_classifiers(random_seed=args.random_state)

        for base_model_name, model in models.items():
            model_name = f"{label_mode}_{base_model_name}"
            results.append(
                train_and_save_model(
                    model_name=model_name,
                    model=model,
                    train_split=mode_train_split,
                    val_split=mode_val_split,
                    full_train_split=mode_full_train_split,
                    model_dir=model_dir,
                    target_id=target_id,
                )
            )

    results.sort(key=lambda result: result["status_accuracy"], reverse=True)

    print("\n=== Validation Comparison ===")
    print("Ranked by status accuracy because 6-class and 4-class class accuracy are not directly comparable.")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['model_name']}: "
            f"status={result['status_accuracy']:.4f}, "
            f"class={result['class_accuracy']:.4f} "
            f"({result['model_path']})"
        )

    print("\n=== Test Reports ===")
    for result in results:
        model_name = result["model_name"]
        label_mode = "four_class" if model_name.startswith("four_class_") else "six_class"
        mode_test_split = relabel_split(test_split, label_mode)
        print(f"\n--- {model_name} ---")
        evaluate_model(result["model"], mode_test_split, "Test")
        test_status_accuracy = evaluate_status_accuracy(result["model"], mode_test_split, target_id)
        print(f"Test status accuracy: {test_status_accuracy:.4f}")

    print("\n=== Training inertial_1d_cnn ===")
    try:
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

        for label_mode in label_modes:
            mode_train_split = relabel_split(inertial_train_split, label_mode)
            mode_val_split = relabel_split(inertial_val_split, label_mode)
            mode_full_train_split = relabel_split(inertial_full_train_split, label_mode)
            mode_test_split = relabel_split(inertial_test_split, label_mode)
            model_name = f"{label_mode}_inertial_1d_cnn"

            cnn_model = InertialCnnClassifier(
                random_seed=args.random_state,
                epochs=args.cnn_epochs,
                batch_size=args.cnn_batch_size,
            )
            result = train_and_save_model(
                model_name=model_name,
                model=cnn_model,
                train_split=mode_train_split,
                val_split=mode_val_split,
                full_train_split=mode_full_train_split,
                model_dir=model_dir,
                target_id=target_id,
            )

            print(f"\n--- {model_name} ---")
            evaluate_model(result["model"], mode_test_split, "Test")
            test_status_accuracy = evaluate_status_accuracy(result["model"], mode_test_split, target_id)
            print(f"Test status accuracy: {test_status_accuracy:.4f}")
    except TorchNotInstalledError as exc:
        print(f"Skipping inertial_1d_cnn: {exc}")


if __name__ == "__main__":
    main()
