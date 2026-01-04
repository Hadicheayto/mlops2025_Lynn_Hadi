from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Dict, Any


class BaseTrainer(ABC):
    """Abstract base class for model training interfaces."""

    @abstractmethod
    def train_linear_regression(self, X_train, y_train) -> BaseEstimator:
        """Train a Linear Regression model."""
        pass

    @abstractmethod
    def train_xgboost(self, X_train, y_train) -> BaseEstimator:
        """Train an XGBoost model."""
        pass

    @abstractmethod
    def evaluate_model(self, model: BaseEstimator, X_eval, y_eval) -> Dict[str, Any]:
        """Evaluate a trained model and return metrics."""
        pass