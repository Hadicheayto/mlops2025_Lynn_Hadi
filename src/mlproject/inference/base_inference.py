from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from typing import Any


class BaseInference(ABC):
    """Abstract base class for model inference."""

    @abstractmethod
    def load_model(self, model_path: Path) -> Any:
        """Load a trained model from file."""
        pass

    @abstractmethod
    def predict(self, model: Any, X: Any) -> Any:
        """Make predictions using the loaded model."""
        pass

    @abstractmethod
    def predict_batch(self, model_path: Path, data_path: Path, output_path: Path) -> None:
        """Run batch inference: load model, load data, predict, save results."""
        pass