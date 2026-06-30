import argparse
from sklearn.metrics import accuracy_score, classification_report

from .model import build_activity_classifier, save_model_to_file
from .data import load_train_val_split, load_test_split

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help='Path to the extracted "UCI HAR Dataset" directory.',
    )
    parser.add_argument(
        "--model-out",
        default="artifacts/activity_classifier.joblib",
    )
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def train_model(train_split, model,  model_save_path="models/activity_classifier.joblib"):
    model.fit(train_split.windows, train_split.labels)
    save_model_to_file(model, model_save_path)


def evaluate_model(model, test_split):
    predictions = model.predict(test_split.windows)
    print(f"Six-class test accuracy: {accuracy_score(test_split.labels, predictions):.4f}")
    print(
        classification_report(
            test_split.labels,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


def main():
    args = parse_args()
    train_split, val_split = load_train_val_split(root_dir=args.data_root, random_state=args.random_state)
    test_split = load_test_split(root_dir=args.data_root)

    model = build_activity_classifier(random_seed=args.random_state)
    train_model(train_split, model, model_save_path=args.model_out)

    evaluate_model(model, test_split)


if __name__ == "__main__":
    main()
