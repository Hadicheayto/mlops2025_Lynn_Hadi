from abc import ABC, abstractmethod
import pandas as pd

class BaseFeaturesComputer(ABC):
    """Interface defining feature engineering steps."""

    @abstractmethod
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
