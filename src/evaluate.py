"""Evaluate the six-class model and the target-conditioned status output."""

import argparse

import numpy as np

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .constants import ACTIVE_ACTIVITIES, STATUS_LABELS, ACTIVITY_NAME_TO_ID
from .data import load_inertial_test_split, load_test_split
from .model import load_model_from_file
from .status_mapping import map_many_to_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--model-path",
        default="artifacts/activity_classifier.joblib",
    )
    parser.add_argument(
        "--target",
        default="WALKING",
        help=(
            "One of WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, "
        ),
    )
    parser.add_argument(
        "--input-kind",
        choices=("features", "inertial"),
        default="features",
        help="Use `features` for 561-feature models and `inertial` for the 1D CNN.",
    )
    return parser.parse_args()


def evaluate_target(y_true_activity: np.ndarray, y_pred_activity: np.ndarray, target: int) -> None:
    y_true_status = map_many_to_status(y_true_activity, target)
    y_pred_status = map_many_to_status(y_pred_activity, target)

    print(f"\n=== Target activity: {target} ===")
    print(
        f"Status accuracy: "
        f"{accuracy_score(y_true_status, y_pred_status):.4f}"
    )
    print(
        classification_report(
            y_true_status,
            y_pred_status,
            labels=list(STATUS_LABELS),
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true_status,
        y_pred_status,
        labels=list(STATUS_LABELS),
    )
    print(
        pd.DataFrame(
            matrix,
            index=[f"true_{label}" for label in STATUS_LABELS],
            columns=[f"pred_{label}" for label in STATUS_LABELS],
        )
    )


def main() -> None:
    args = parse_args()
    if args.input_kind == "inertial":
        test = load_inertial_test_split(root_dir=args.data_dir)
    else:
        test = load_test_split(root_dir=args.data_dir)
    model = load_model_from_file(args.model_path)
    activity_predictions = model.predict(test.windows)

    print(
        f"Six-class activity accuracy: "
        f"{accuracy_score(test.labels, activity_predictions):.4f}"
    )
    print(
        classification_report(
            test.labels,
            activity_predictions,
            digits=4,
            zero_division=0,
        )
    )

    target = args.target.strip().upper()
    if target not in ACTIVE_ACTIVITIES:
        raise ValueError(
            f"Invalid target: {args.target}. Must be one of "
            f"{list(ACTIVE_ACTIVITIES)}"
        )

    evaluate_target(test.labels, activity_predictions, ACTIVITY_NAME_TO_ID[target])


if __name__ == "__main__":
    main()
