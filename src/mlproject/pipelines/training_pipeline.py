# ========================
# 1. Imports
# ========================
import os
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

# ========================
# 2. AWS Config
# ========================
role = os.environ["SAGEMAKER_ROLE_ARN"]
session = PipelineSession()
bucket = sagemaker.Session().default_bucket()

# ========================
# 3. Pipeline Parameters
# ========================
train_data_s3 = ParameterString(
    name="TrainDataS3",
    default_value=f"s3://{bucket}/data/train"
)

target_column = ParameterString(
    name="TargetColumn",
    default_value="trip_duration"
)

# ========================
# 4. Preprocessing + Feature Engineering
# ========================
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session,
)

preprocess_step = ProcessingStep(
    name="PreprocessAndFeatures",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=train_data_s3,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/data/processed/train",
        )
    ],
    code="scripts/preprocess.py",
    job_arguments=[
        "--input_path", "/opt/ml/processing/input/train.csv",
        "--output_path", "/opt/ml/processing/output/train.csv",
        "--target", target_column,
    ],
)

# ========================
# 5. Training Step
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
    name="ModelTraining",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocess_step.properties
            .ProcessingOutputConfig.Outputs["train"]
            .S3Output.S3Uri,
            content_type="text/csv",
        )
    },
)

# ========================
# 6. Pipeline Definition
# ========================
pipeline = Pipeline(
    name="TaxiTrainingPipeline",
    parameters=[train_data_s3, target_column],
    steps=[preprocess_step, train_step],
    sagemaker_session=session,
)
