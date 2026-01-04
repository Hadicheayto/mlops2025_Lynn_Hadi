from sklearn.feature_extraction import DictVectorizer
from pathlib import Path
import pickle

class TransformersManager:
    """
    Handles feature encoding and persistence using DictVectorizer
    """

    def __init__(self):
        self.dv = DictVectorizer(sparse=True)

    # -----------------------------
    # Fit AND save in a single step
    # -----------------------------
    def fit_and_save(self, df, categorical_cols, numerical_cols, output_dir="src/mlproject/data/transformers"):
        """
        Fit the DictVectorizer on df and immediately save it to disk.
        """
        df = df.copy()
        df[categorical_cols] = df[categorical_cols].astype(str)

        feature_dicts = df[categorical_cols + numerical_cols].to_dict(orient="records")
        self.dv.fit(feature_dicts)

        # Save immediately
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "dict_vectorizer.pkl", "wb") as f:
            pickle.dump(self.dv, f)

        print(f"[INFO] DictVectorizer fitted and saved to {output_dir}")

        return self.dv

    # -----------------------------
    # Transform using already fitted DV
    # -----------------------------
    def transform(self, df, categorical_cols, numerical_cols):
        df = df.copy()
        df[categorical_cols] = df[categorical_cols].astype(str)
        feature_dicts = df[categorical_cols + numerical_cols].to_dict(orient="records")
        return self.dv.transform(feature_dicts)

    # -----------------------------
    # Transform taxi data (predefined columns)
    # -----------------------------
    def transform_taxi_data(self, df):
        """
        Transform taxi trip data using predefined columns.
        """
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
        return self.transform(df, categorical_cols, numerical_cols)

    # -----------------------------
    # Static load method
    # -----------------------------
    @staticmethod
    def load(path="src/mlproject/data/transformers/dict_vectorizer.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)