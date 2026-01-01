from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing."""

    @abstractmethod
    def remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with null values."""
        pass

    @abstractmethod
    def remove_invalid_passengers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where passenger_count <= 0."""
        pass

    @abstractmethod
    def add_trip_duration_minutes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert trip_duration from seconds to minutes"""
        pass


    @abstractmethod
    def remove_duration_outliers(
        self,
        df: pd.DataFrame,
        min_minutes: float,
        max_minutes: float
    ) -> pd.DataFrame:
        """Remove trips with unrealistic duration values"""
        pass