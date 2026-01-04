import pandas as pd
import numpy as np

from .base_featurise import BaseFeaturesComputer

class Featurizer(BaseFeaturesComputer):
    """
    Stateless feature engineering class.
    Creates time-based and distance-based features
    and removes leakage / unused columns.
    """

    def __init__(self):
        self.categorical_cols = [
            "vendor_id",
            "store_and_fwd_flag",
        ]

        self.columns_to_drop = [
            "id",
            "pickup_datetime",
            "dropoff_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]

    # -----------------------------
    # Time-based features
    # -----------------------------
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
        df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
        df["is_night"] = df["pickup_hour"].between(0, 6).astype(int)

        return df

    # -----------------------------
    # Distance-based features
    # -----------------------------
    def add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["trip_distance_km"] = self._haversine(
            df["pickup_longitude"],
            df["pickup_latitude"],
            df["dropoff_longitude"],
            df["dropoff_latitude"],
        )

        df["manhattan_distance"] = (
            abs(df["pickup_latitude"] - df["dropoff_latitude"]) +
            abs(df["pickup_longitude"] - df["dropoff_longitude"])
        ) * 111

        return df

    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371  # Earth radius in km
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )

        return 2 * R * np.arcsin(np.sqrt(a))

    # -----------------------------
    # Categorical preparation
    # -----------------------------
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.categorical_cols] = df[self.categorical_cols].astype(str)
        return df

    # -----------------------------
    # Drop leakage / unused columns
    # -----------------------------
    def drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df.drop(columns=self.columns_to_drop, errors="ignore")

    # -----------------------------
    # Orchestrator
    # -----------------------------
    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.add_time_features(df)
        df = self.add_distance_features(df)
        df = self.encode_categorical_features(df)
        df = self.drop_unused_columns(df)
        return df
