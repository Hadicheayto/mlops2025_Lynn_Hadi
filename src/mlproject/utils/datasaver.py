from pathlib import Path
import pandas as pd


def save_dataframe(
    df: pd.DataFrame,
    output_path: str,
    index: bool = False
) -> None:
    """
    Save a DataFrame to CSV, ensuring the output directory exists.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    output_path : str
        Destination CSV path
    index : bool, optional
        Whether to write row index, by default False
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=index)