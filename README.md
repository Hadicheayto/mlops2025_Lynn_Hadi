# mlops2025_Lynn_Hadi

<details>
<summary><strong>📁 Project Structure</strong></summary>

Below is the current project layout:

```text
mlops2025_Lynn_Hadi/

# ── Project metadata
├── README.md
├── pyproject.toml
├── main.py

# ── Configuration & experiments
├── configs/
├── notebooks/

# ── Entry points
├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── batch_inference.py

# ── Core package (src layout)
├── src/
│   └── mlproject/
│       ├── __init__.py
│       ├── data/
│       ├── preprocess/
│       ├── features/
│       ├── train/
│       ├── inference/
│       ├── pipelines/
│       └── utils/

# ── Testing
└── tests/
