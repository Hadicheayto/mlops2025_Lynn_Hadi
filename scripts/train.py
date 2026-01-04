import argparse
import sys
from pathlib import Path
import pickle
import json

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlproject.features.TransformersManager import TransformersManager
from src.mlproject.train.trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train model and evaluate on eval set")

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["linear", "xgboost"], default="xgboost")

    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_experiment("mlops_taxi_trip_duration")

    with mlflow.start_run(run_name=args.model_type):
        mlflow.log_param("model_type", args.model_type)

        # Load transformers
        dv = TransformersManager.load(Path("src/mlproject/data/transformers/dict_vectorizer.pkl"))

        # Load train/eval splits
        X_train = pd.read_csv(Path(args.train_dir) / "X_train.csv")
        y_train = pd.read_csv(Path(args.train_dir) / "y_train.csv").values.ravel()

        X_eval = pd.read_csv(Path(args.eval_dir) / "X_eval.csv")
        y_eval = pd.read_csv(Path(args.eval_dir) / "y_eval.csv").values.ravel()

        # Initialize trainer
        trainer = ModelTrainer(dv)

        transformers_manager = TransformersManager()
        transformers_manager.dv = dv

        X_train_transformed = transformers_manager.transform_taxi_data(X_train)
        X_eval_transformed = transformers_manager.transform_taxi_data(X_eval)

        # Train model
        if args.model_type == "linear":
            model = trainer.train_linear_regression(X_train_transformed, y_train)
            model_name = "Linear Regression"
        elif args.model_type == "xgboost":
            model = trainer.train_xgboost(X_train_transformed, y_train)
            model_name = "XGBoost"
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        # Evaluate model
        metrics = trainer.evaluate_model(model, X_eval_transformed, y_eval)

        # -----------------------------
        # MLflow logging
        # -----------------------------
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])

        # Save model locally (unchanged behavior)
        model_output_path = Path(args.model_output)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)

        dv_path = model_output_path.with_suffix(".dv.pkl")
        with open(dv_path, "wb") as f:
            pickle.dump(dv, f)

        # Save metrics (convert numpy types to native Python types)
        metrics_path = model_output_path.with_suffix(".metrics.json")
        metrics_serializable = {}
        for k, v in metrics.items():
            if hasattr(v, 'tolist'):  # numpy array or scalar
                metrics_serializable[k] = v.tolist()
            else:
                metrics_serializable[k] = v
        with open(metrics_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)

        # Log artifacts to MLflow
        mlflow.log_artifact(model_output_path)
        mlflow.log_artifact(dv_path)
        mlflow.log_artifact(metrics_path)

        # Optional: register sklearn-compatible model
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
