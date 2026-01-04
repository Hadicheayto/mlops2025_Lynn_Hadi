import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Any

from .base_inference import BaseInference
from ..features.featurizer import Featurizer
from ..features.TransformersManager import TransformersManager


class Inference(BaseInference):
    """Concrete implementation for model inference operations."""

    def __init__(self):
        self.featurizer = Featurizer()

    def load_model(self, model_path: Path) -> Any:
        """Load a trained model from pickle file."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, model: Any, X: Any) -> Any:
        """Make predictions using the loaded model."""
        return model.predict(X)

    def predict_batch(self, model_path: Path, data_path: Path, output_path: Path) -> None:
        """
        Run complete batch inference pipeline:
        1. Load model
        2. Load and featurize data
        3. Transform features
        4. Make predictions
        5. Save results
        """
        # Load model
        print(f"Loading model from: {model_path}")
        model = self.load_model(model_path)

        # Load test data
        print(f"Loading test data from: {data_path}")
        test_df = pd.read_csv(data_path)

        # Save IDs for output (featurizer will drop them)
        test_ids = test_df['id'].copy()

        # Featurize test data
        print("Featurizing test data...")
        test_df = self.featurizer.featurize(test_df)

        # Load and setup transformers
        dv = TransformersManager.load(Path("src/mlproject/data/transformers/dict_vectorizer.pkl"))
        transformers_manager = TransformersManager()
        transformers_manager.dv = dv

        # Transform features
        print("Transforming test features...")
        X_test_transformed = transformers_manager.transform_taxi_data(test_df)
        print(f"Test features shape: {X_test_transformed.shape}")

        # Make predictions
        print("Making predictions...")
        predictions = self.predict(model, X_test_transformed)

        # Create output DataFrame
        output_df = pd.DataFrame({
            'id': test_ids,
            'prediction': predictions
        })

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")