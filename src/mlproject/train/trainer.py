import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from .base_train import BaseTrainer


class ModelTrainer(BaseTrainer):
    """Interface for training and evaluating regression models."""

    def __init__(self, dv):
        """
        Initialize the trainer with a fitted DictVectorizer.

        Args:
            dv: Fitted DictVectorizer for feature transformation
        """
        self.dv = dv

    def train_linear_regression(self, X_train_transformed, y_train):
        """
        Train a Linear Regression model.

        Args:
            X_train_transformed: Transformed training features
            y_train: Training targets

        Returns:
            Trained LinearRegression model
        """
        model = LinearRegression()
        model.fit(X_train_transformed, y_train)
        return model

    def train_xgboost(self, X_train_transformed, y_train):
        """
        Train an XGBoost model.

        Args:
            X_train_transformed: Transformed training features
            y_train: Training targets

        Returns:
            Trained XGBRegressor model
        """
        model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model.fit(X_train_transformed, y_train)
        return model

    def evaluate_model(self, model, X_eval_transformed, y_eval):
        """
        Evaluate a trained model.

        Args:
            model: Trained model
            X_eval_transformed: Transformed evaluation features
            y_eval: Evaluation targets

        Returns:
            dict: Dictionary with RMSE, MAE, R2 metrics
        """
        y_pred = model.predict(X_eval_transformed)

        rmse = root_mean_squared_error(y_eval, y_pred)
        mae = mean_absolute_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "predictions": y_pred
        }