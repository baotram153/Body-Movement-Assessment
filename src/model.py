from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import joblib


def build_activity_classifier(random_seed: int = 2026) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                random_state=random_seed, 
                max_iter=1000,
                verbose=1
                )
            ),
        ]
    )

def load_model_from_file(file_path: str) -> Pipeline:
    model_path = Path(file_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def save_model_to_file(model: Pipeline, file_path: str) -> None:
    model_path = Path(file_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
