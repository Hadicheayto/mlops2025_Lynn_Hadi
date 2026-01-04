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

# -------------------------
# Full pipeline
# -------------------------
all: preprocess features train


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
# Batch inference (optional)
# -------------------------
inference:
	python scripts/batch_inference.py
