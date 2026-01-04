import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mlproject.utils.dataloader import DataLoader
from mlproject.preprocess.preprocessor import Preprocess
from mlproject.utils.datasaver import save_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing step")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_train", type=str, required=True)
    parser.add_argument("--output_test", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------
    # Load data
    # ------------------
    loader = DataLoader(
        train_path=args.train_path,
        test_path=args.test_path,
    )
    train_df, test_df = loader.load()

    # ------------------
    # Preprocess
    # ------------------
    preprocessor = Preprocess()

    train_df = preprocessor.remove_nulls(train_df)
    train_df = preprocessor.remove_invalid_passengers(train_df)
    train_df = preprocessor.add_trip_duration_minutes(train_df)
    train_df = preprocessor.remove_duration_outliers(train_df)

    test_df = preprocessor.remove_nulls(test_df)
    test_df = preprocessor.remove_invalid_passengers(test_df)

    # ------------------
    # Save outputs
    # ------------------
    save_dataframe(train_df, args.output_train)
    save_dataframe(test_df, args.output_test)

    print("✅ Preprocessing completed successfully")
    print(f"📁 Train saved to: {args.output_train}")
    print(f"📁 Test saved to: {args.output_test}")


if __name__ == "__main__":
    main()
