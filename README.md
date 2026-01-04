# mlops2025_Lynn_Hadi

## Overview
End-to-end MLOps pipeline for taxi trip duration prediction, covering
data preprocessing, feature engineering, model training, evaluation,
and experiment tracking.

---
## Project Structure

```
mlops2025_Lynn_Hadi/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py

├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── batch_inference.py

├── src/
│   ├── mlproject/
│   │   ├── __init__.py
│   │   ├── data/
│   │   ├── preprocess/
│   │   ├── features/
│   │   ├── train/
│   │   ├── inference/
│   │   ├── pipelines/
│   │   └── utils/
│   └── mlproject.egg-info/

├── configs/
└── tests/
```
---
## Pipeline Execution (Makefile)


This project uses a Makefile to orchestrate the ML pipeline.

### Available Commands
```bash
make preprocess
make features
make train
make batch_inference
```
---

## Experiment Tracking (MLflow)
This project uses MLflow to track and compare machine learning experiments in a reproducible way.

```md
What is tracked:

- Models: Linear Regression, XGBoost
- Metrics: RMSE, MAE, R²

```
How it works

MLflow is integrated directly into the training script (train.py).
Each model run is logged as a separate MLflow run under the same experiment.
MLflow enables reproducibility, model comparison, and experiment auditing
without modifying the core training logic.


## Launch MLflow UI
```bash
uv run mlflow ui
```

then open: 
```bash
http://127.0.0.1:5000
```
## Results

MLflow enables easy comparison between models.
In this project, XGBoost outperforms Linear Regression across all evaluation metrics.

![MLflow experiment comparison](https://github.com/user-attachments/assets/9e78e3c7-e28c-4b54-b526-381e79318b28)


