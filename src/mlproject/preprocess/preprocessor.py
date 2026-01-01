import pandas as pd
from .base_preprocessor import BasePreprocessor


class Preprocess(BasePreprocessor):
    """Concrete preprocessing implementation."""

    def remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing any null values.
        """
        return df.dropna()

    def remove_invalid_passengers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where passenger_count is less than or equal to 0.
        """
        return df[df["passenger_count"] > 0]

    def add_trip_duration_minutes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert trip_duration from seconds to minutes (in place).
        """
        if "trip_duration" not in df.columns:
            raise ValueError("trip_duration column not found in dataframe")

        df = df.copy()
        df["trip_duration"] = df["trip_duration"] / 60
        return df

    def remove_duration_outliers(
        self,
        df: pd.DataFrame,
        min_minutes: float = 1,
        max_minutes: float = 60
    ) -> pd.DataFrame:
        """
        Remove trips with unrealistic trip_duration values (in minutes).
        """
        if "trip_duration" not in df.columns:
            raise ValueError(
                "trip_duration column not found. "
                "Call add_trip_duration_minutes first."
            )

        return df[
            (df["trip_duration"] >= min_minutes) &
            (df["trip_duration"] <= max_minutes)
        ].copy()