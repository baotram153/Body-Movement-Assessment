from __future__ import annotations

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class InertialCnnClassifier:
    def __init__(
        self,
        random_seed: int = 2026,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        device: str | None = None,
    ) -> None:
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model_ = None
        self.input_channels_ = None
        self.state_dict_ = None
        self.classes_ = np.arange(1, 7, dtype=np.int64)

    def _build_model(self, input_channels: int, class_count: int) -> nn.Module:
        class SmallInertialCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),

                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),

                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, class_count),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return SmallInertialCnn()

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_seed)

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(np.asarray(y, dtype=np.int64) - 1, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.input_channels_ = X.shape[1]
        self.model_ = self._build_model(input_channels=self.input_channels_, class_count=len(self.classes_)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            correct = 0
            seen = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                seen += batch_X.size(0)

            print(
                f"cnn epoch {epoch:02d}/{self.epochs} "
                f"loss={total_loss / seen:.4f} accuracy={correct / seen:.4f}"
            )

        self.state_dict_ = {
            key: value.detach().cpu()
            for key, value in self.model_.state_dict().items()
        }
        return self

    def _ensure_model_loaded(self) -> None:
        if self.model_ is not None:
            return
        if self.state_dict_ is None or self.input_channels_ is None:
            raise ValueError("Model has not been fitted.")

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_ = self._build_model(
            input_channels=self.input_channels_,
            class_count=len(self.classes_),
        ).to(device)
        self.model_.load_state_dict(self.state_dict_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_model_loaded()

        device = next(self.model_.parameters()).device
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        self.model_.eval()
        with torch.no_grad():
            for (batch_X,) in loader:
                logits = self.model_(batch_X.to(device))
                predictions.append(logits.argmax(dim=1).cpu().numpy() + 1)

        return np.concatenate(predictions).astype(np.int32)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state["state_dict_"] = {
                key: value.detach().cpu()
                for key, value in self.model_.state_dict().items()
            }
        state["model_"] = None
        return state
