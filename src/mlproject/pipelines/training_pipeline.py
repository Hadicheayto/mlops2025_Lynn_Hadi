# ========================
# 1. Imports
# ========================
import os
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

# ========================
# 2. AWS Config
# ========================
role = os.environ["SAGEMAKER_ROLE_ARN"]
session = PipelineSession()
bucket = "lynn-hadi-taxi-mlops"   # 🔴 DO NOT use default_bucket()

# ========================
# 3. Preprocessing Step
# ========================
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session,
)

preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/data/raw/",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/data/processed/",
        )
    ],
    code="scripts/preprocess.py",
    job_arguments=[
        "--train_path", "/opt/ml/processing/input/train.csv",
        "--test_path", "/opt/ml/processing/input/test.csv",
        "--output_train", "/opt/ml/processing/output/train.csv",
        "--output_test", "/opt/ml/processing/output/test.csv",
    ],
)

# ========================
# 4. Training Step
# ========================
estimator = SKLearn(
    entry_point="scripts/train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/data/processed/train.csv",
            content_type="text/csv",
        )
    },
)

# ========================
# 5. Pipeline Definition
# ========================
pipeline = Pipeline(
    name="TaxiTrainingPipeline",
    steps=[preprocess_step, train_step],
    sagemaker_session=session,
)
