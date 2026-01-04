.PHONY: preprocess features train inference all

# Paths
RAW_TRAIN=src/mlproject/data/train.csv
RAW_TEST=src/mlproject/data/test.csv

CLEAN_TRAIN=src/mlproject/data/processed/train_clean.csv
CLEAN_TEST=src/mlproject/data/processed/test_clean.csv

TRAIN_SPLIT=src/mlproject/data/splits/train
EVAL_SPLIT=src/mlproject/data/splits/eval


MODEL_OUTPUT=models/trained_model.pkl
MODEL_TYPE=xgboost
OUTPUT_DIR=outputs

# -------------------------
# Full pipeline (preprocessing + features + training)
# -------------------------
all: preprocess features train


# -------------------------
# Complete pipeline (including inference)
# -------------------------
full: preprocess features train inference

# -------------------------
# Complete pipeline with MLflow logging
# -------------------------
full-mlflow: preprocess features train-mlflow inference


# -------------------------
# Preprocessing
# -------------------------
preprocess:
	python scripts/preprocess.py \
		--train_path $(RAW_TRAIN) \
		--test_path $(RAW_TEST) \
		--output_train $(CLEAN_TRAIN) \
		--output_test $(CLEAN_TEST)


# -------------------------
# Feature engineering + split
# -------------------------
features:
	python scripts/feature_engineering.py \
		--train_input $(CLEAN_TRAIN) \
		--test_input $(CLEAN_TEST) \
		--train_dir $(TRAIN_SPLIT) \
		--eval_dir $(EVAL_SPLIT)


# -------------------------
# Training
# -------------------------
train:
	python scripts/train.py \
		--train_dir $(TRAIN_SPLIT) \
		--eval_dir $(EVAL_SPLIT) \
		--model_output $(MODEL_OUTPUT) \
		--model_type $(MODEL_TYPE)

# -------------------------
# Training with MLflow logging
# -------------------------
train-mlflow:
	python scripts/train.py \
		--train_dir $(TRAIN_SPLIT) \
		--eval_dir $(EVAL_SPLIT) \
		--model_output $(MODEL_OUTPUT) \
		--model_type $(MODEL_TYPE) \
		--use_mlflow

# -------------------------
# Batch inference
# -------------------------
inference:
	python scripts/batch_inference.py \
		--model_path $(MODEL_OUTPUT) \
		--test_data $(RAW_TEST) \
		--output_dir $(OUTPUT_DIR)

# -------------------------
# Clean generated files
# -------------------------
clean:
	rm -rf outputs/* models/* src/mlproject/data/processed/* src/mlproject/data/splits/* src/mlproject/data/transformers/*

# -------------------------
# Help
# -------------------------
help:
	@echo "Available targets:"
	@echo "  all         - Run preprocessing, features, and training"
	@echo "  full        - Run complete pipeline including inference"
	@echo "  full-mlflow - Run complete pipeline with MLflow logging"
	@echo "  preprocess  - Clean and preprocess data"
	@echo "  features    - Create features and data splits"
	@echo "  train       - Train the model (no MLflow)"
	@echo "  train-mlflow- Train the model with MLflow logging"
	@echo "  inference   - Run batch inference on test data"
	@echo "  clean       - Remove generated files"
