import argparse
import sys
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
import json

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlproject.features.TransformersManager import TransformersManager
from src.mlproject.train.trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train model and evaluate on eval set")

    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to evaluation data directory")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--model_type", type=str, choices=["linear", "xgboost"], default="xgboost", help="Type of model to train")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load transformers
    dv = TransformersManager.load(Path("src/mlproject/data/transformers/dict_vectorizer.pkl"))

    # Load train/eval splits
    X_train = pd.read_csv(Path(args.train_dir) / "X_train.csv")
    y_train = pd.read_csv(Path(args.train_dir) / "y_train.csv").values.ravel()

    X_eval = pd.read_csv(Path(args.eval_dir) / "X_eval.csv")
    y_eval = pd.read_csv(Path(args.eval_dir) / "y_eval.csv").values.ravel()

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Initialize trainer
    trainer = ModelTrainer(dv)

    # Transform features using TransformersManager
    transformers_manager = TransformersManager()
    transformers_manager.dv = dv  # Set the loaded dv

    X_train_transformed = transformers_manager.transform_taxi_data(X_train)
    X_eval_transformed = transformers_manager.transform_taxi_data(X_eval)

    print(f"Transformed training features shape: {X_train_transformed.shape}")

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

    print(f"\n{model_name} Evaluation Results:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")

    # Save model
    model_output_path = Path(args.model_output)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {args.model_output}")

    # Persist artifacts
    # -----------------------------
    # Save model
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    # Save DictVectorizer alongside model
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
    print("\n💾 Artifacts saved:")
    print(f"- Model          → {model_output_path}")
    print(f"- DictVectorizer → {dv_path}")
    print(f"- Metrics        → {metrics_path}")
if __name__ == "__main__":
    main()


