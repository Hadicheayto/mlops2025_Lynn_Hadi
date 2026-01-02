import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlproject.features.featurizer import Featurizer
from src.mlproject.features.TransformersManager import TransformersManager
from src.mlproject.utils.dataloader import DataLoader
from src.mlproject.features.split import split_train_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Featurize data, split train/eval, and save outputs."
    )

    parser.add_argument("--train_input", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--test_input", type=str, required=True, help="Path to test CSV")

    parser.add_argument("--train_dir", type=str, required=True, help="Directory to save train split")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory to save eval split")

    args = parser.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    loader = DataLoader(
        train_path=args.train_input,
        test_path=args.test_input,
    )
    df_train, df_test = loader.load()

    print(f"[INFO] Loaded {len(df_train)} train rows and {len(df_test)} test rows")

    # -----------------------------
    # Featurization
    # -----------------------------
    featurizer = Featurizer()

    df_train_feat = featurizer.featurize(df_train)
    df_test_feat = featurizer.featurize(df_test)

    print("[INFO] Featurization completed")

    # -----------------------------
    # Split train / eval
    # -----------------------------
    X_train, y_train, X_eval, y_eval = split_train_eval(df_train_feat)

    print(
        f"[INFO] Split data → "
        f"Train: {len(X_train)} rows | Eval: {len(X_eval)} rows"
    )

    # -----------------------------
    # Save train split
    # -----------------------------
    train_dir = Path(args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(train_dir / "X_train.csv", index=False)
    y_train.to_csv(train_dir / "y_train.csv", index=False)

    print(
        f"[INFO] Training data saved to "
        f"{train_dir / 'X_train.csv'} and {train_dir / 'y_train.csv'}"
    )

    # -----------------------------
    # Save eval split
    # -----------------------------
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    X_eval.to_csv(eval_dir / "X_eval.csv", index=False)
    y_eval.to_csv(eval_dir / "y_eval.csv", index=False)

    print(
        f"[INFO] Evaluation data saved to "
        f"{eval_dir / 'X_eval.csv'} and {eval_dir / 'y_eval.csv'}"
    )

    # -----------------------------
    # Fit DictVectorizer on TRAIN only
    # -----------------------------
    categorical_cols = ["vendor_id", "store_and_fwd_flag"]
    numerical_cols = [
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "is_weekend",
        "trip_distance_km",
        "manhattan_distance",
        "is_night",
    ]

    tm = TransformersManager()
    tm.fit_and_save(
        X_train,
        categorical_cols,
        numerical_cols,
        output_dir="src/mlproject/data/transformers",
    )

    print("[INFO] DictVectorizer fitted on TRAIN only and saved")

    print("[INFO] Feature engineering + split pipeline completed successfully")


#python scripts/feature_engineering.py --train_input src/mlproject/data/processed/train_clean.csv --test_input src/mlproject/data/processed/test_clean.csv --train_output src/mlproject/data/featurized/train_features.csv --test_output src/mlproject/data/featurized/test_features.csv
