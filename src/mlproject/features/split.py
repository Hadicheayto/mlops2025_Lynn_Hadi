import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_train_eval(
    df: pd.DataFrame,
    target_col: str = "trip_duration",
    eval_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split dataframe into X_train, y_train, X_eval, y_eval.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe after preprocessing and feature engineering.
    target_col : str
        Name of the target column.
    eval_size : float
        Fraction of data to use for evaluation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_eval : pd.DataFrame
    y_eval : pd.Series
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train / evaluation split
    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=eval_size,
        random_state=random_state,
        shuffle=True,
    )

    return X_train, y_train, X_eval, y_eval
