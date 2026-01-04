import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlproject.inference import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference on test data")

    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model pickle file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data CSV file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for predictions")

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate output filename with today's date
    today = datetime.now().strftime("%Y%m%d")
    output_file = Path(args.output_dir) / f"{today}_predictions.csv"

    # Initialize inference engine
    inference = Inference()

    # Run batch inference
    inference.predict_batch(
        model_path=Path(args.model_path),
        data_path=Path(args.test_data),
        output_path=output_file
    )


if __name__ == "__main__":
    main()
