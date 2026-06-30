from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

FEATURE_COUNT = 561
SIGNAL_WINDOW_SIZE = 128
SAMPLE_RATE_HZ = 50
DEFAULT_STRIDE = 64

@dataclass
class DatasetSplit:
    # these data must be contained in train/val/test splits
    windows: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray


def _read_signal(split_dir: Path, split: str, channel: str) -> np.ndarray:
    signal_path = (
        split_dir
        / "Inertial Signals"
        / f"{channel}_{split}.txt"
    )

    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")
    
    values = np.loadtxt(signal_path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != SIGNAL_WINDOW_SIZE:
        raise ValueError(f"Expected 2D array with shape (n_samples, {SIGNAL_WINDOW_SIZE}), got {values.shape}")
    return values


def _load_split(data_dir: Path, split: str) -> DatasetSplit:
    X_path = data_dir / f"X_{split}.txt"
    y_path = data_dir / f"y_{split}.txt"
    subjects_path = data_dir / f"subject_{split}.txt"

    if not X_path.exists():
        raise FileNotFoundError(f"File not found: {X_path}")
    
    X = np.loadtxt(X_path, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Expected 2D array with shape (n_samples, {FEATURE_COUNT}), got {X.shape}")
    
    if not y_path.exists():
        raise FileNotFoundError(f"File not found: {y_path}")
    
    y = np.loadtxt(y_path, dtype=np.int32)

    if not subjects_path.exists():
        raise FileNotFoundError(f"File not found: {subjects_path}")
    subjects = np.loadtxt(subjects_path, dtype=np.int32)

    return DatasetSplit(windows=X, labels=y, subjects=subjects)


def load_train_val_split(root_dir: Path, train_data_dir: Path = Path("train"), random_state: int = 2026) -> Tuple[DatasetSplit, DatasetSplit]:
    root_path = Path(root_dir)
    data_path = root_path / train_data_dir
    train_split = _load_split(data_path, split="train")

    X_train, X_val, y_train, y_val, subjects_train, subjects_val = train_test_split(
        train_split.windows, train_split.labels, train_split.subjects, test_size=0.2, random_state=random_state
    )

    reduced_train_split = DatasetSplit(windows=X_train, labels=y_train, subjects=subjects_train)
    val_split = DatasetSplit(windows=X_val, labels=y_val, subjects=subjects_val)
    return reduced_train_split, val_split


def load_test_split(root_dir: Path, test_data_dir: Path = Path("test")) -> DatasetSplit:
    root_path = Path(root_dir)
    data_path = root_path / test_data_dir
    test_split = _load_split(data_path, split="test")
    return test_split
