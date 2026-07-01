"""Evaluate the six-class model and the target-conditioned status output."""

import argparse
from time import perf_counter

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
    parser.add_argument(
        "--latency-batch-size",
        type=int,
        default=1,
        help="Number of samples per latency timing call.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=5,
        help="Number of untimed warmup prediction calls.",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=50,
        help="Number of timed prediction calls.",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Disable latency measurement.",
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


def evaluate_latency(
    model,
    windows: np.ndarray,
    target: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("--latency-batch-size must be greater than 0")
    if warmup < 0:
        raise ValueError("--latency-warmup must be greater than or equal to 0")
    if repeats <= 0:
        raise ValueError("--latency-repeats must be greater than 0")

    actual_batch_size = min(batch_size, len(windows))
    batch = windows[:actual_batch_size]

    for _ in range(warmup):
        activity_predictions = model.predict(batch)
        map_many_to_status(activity_predictions, target)

    durations_ms = []
    for _ in range(repeats):
        start = perf_counter()
        activity_predictions = model.predict(batch)
        map_many_to_status(activity_predictions, target)
        durations_ms.append((perf_counter() - start) * 1000)

    durations = np.array(durations_ms)
    print("\n=== Pipeline Latency (model prediction + status mapping)===")
    print(f"Batch size: {actual_batch_size}")
    print(f"Warmup runs: {warmup}")
    print(f"Timed runs: {repeats}")
    print(f"Mean batch latency: {durations.mean():.4f} ms")
    print(f"Median batch latency: {np.median(durations):.4f} ms")
    print(f"P95 batch latency: {np.percentile(durations, 95):.4f} ms")
    print(f"Mean per-sample latency: {durations.mean() / actual_batch_size:.4f} ms")


def main() -> None:
    args = parse_args()
    if args.input_kind == "inertial":
        test = load_inertial_test_split(root_dir=args.data_dir)
    else:
        test = load_test_split(root_dir=args.data_dir)
    model = load_model_from_file(args.model_path)
    activity_predictions = model.predict(test.windows)

    target = args.target.strip().upper()
    if target not in ACTIVE_ACTIVITIES:
        raise ValueError(
            f"Invalid target: {args.target}. Must be one of "
            f"{list(ACTIVE_ACTIVITIES)}"
        )
    target_id = ACTIVITY_NAME_TO_ID[target]

    if not args.skip_latency:
        evaluate_latency(
            model=model,
            windows=test.windows,
            target=target_id,
            batch_size=args.latency_batch_size,
            warmup=args.latency_warmup,
            repeats=args.latency_repeats,
        )

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

    evaluate_target(test.labels, activity_predictions, target_id)


if __name__ == "__main__":
    main()
