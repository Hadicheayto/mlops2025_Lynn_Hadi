# src/mlproject/pipelines/training_pipeline.py

# ========================
# 1. Imports
# ========================
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator

# ========================
# 2. AWS Session & Config
# ========================
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name()

# ========================
# 3. ScriptProcessor
# (Used for preprocess + feature engineering)
# ========================
processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
    ),
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
)

# ========================
# 4. Preprocessing Step
# ========================
preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/data/raw",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/data/clean",
        )
    ],
    code="../scripts/preprocess.py",
    job_arguments=[
        "--train_path", "/opt/ml/processing/input/train.csv",
        "--test_path", "/opt/ml/processing/input/test.csv",
        "--output_train", "/opt/ml/processing/output/train.csv",
        "--output_test", "/opt/ml/processing/output/test.csv",
    ],
)

# ========================
# 5. Feature Engineering Step
# ========================
feature_step = ProcessingStep(
    name="FeatureEngineering",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=preprocess_step.properties
            .ProcessingOutputConfig.Outputs["output"]
            .S3Output.S3Uri,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/train",
            destination=f"s3://{bucket}/data/train",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/eval",
            destination=f"s3://{bucket}/data/eval",
        ),
    ],
    code="../scripts/feature_engineering.py",
    job_arguments=[
        "--train_input", "/opt/ml/processing/input/train.csv",
        "--test_input", "/opt/ml/processing/input/test.csv",
        "--train_dir", "/opt/ml/processing/train",
        "--eval_dir", "/opt/ml/processing/eval",
    ],
)

# ========================
# 6. Training Step
# ========================
estimator = Estimator(
    image_uri=processor.image_uri,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    entry_point="../scripts/train.py",
    output_path=f"s3://{bucket}/models/taxi_model",  # save trained model here
    hyperparameters={
        "model_type": "xgboost",  # default model type
    }
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": feature_step.properties
        .ProcessingOutputConfig.Outputs["train"]
        .S3Output.S3Uri,
        "eval": feature_step.properties
        .ProcessingOutputConfig.Outputs["eval"]
        .S3Output.S3Uri,
    },
)

# ========================
# 7. Pipeline Definition
# ========================
pipeline = Pipeline(
    name="MLTrainingPipeline",
    steps=[
        preprocess_step,
        feature_step,
        train_step,
    ],
    sagemaker_session=session,
)

# Create or update the pipeline in SageMaker
pipeline.upsert(role_arn=role)

# Start the pipeline execution
pipeline.start()

print("✅ Training pipeline started successfully!")
