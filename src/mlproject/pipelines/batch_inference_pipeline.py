# ========================
# 1. Imports (CORRECT)
# ========================
import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TransformInput

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.transformer import Transformer

# ========================
# 2. AWS Config (SAFE)
# ========================
region = sagemaker.Session().boto_region_name
role = os.environ["SAGEMAKER_ROLE_ARN"]
session = PipelineSession()
bucket = sagemaker.Session().default_bucket()

# ========================
# 3. Pipeline Parameters
# ========================
raw_data_s3 = ParameterString(
    name="RawDataS3",
    default_value=f"s3://{bucket}/data/raw"
)

model_artifact_s3 = ParameterString(
    name="ModelArtifactS3"
)

output_s3 = ParameterString(
    name="PredictionsS3",
    default_value=f"s3://{bucket}/predictions"
)

# ========================
# 4. Processor (preprocess + features)
# ========================
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session,
)

# ========================
# 5. Preprocessing + Feature Engineering
# ========================
preprocess_step = ProcessingStep(
    name="PreprocessAndFeatures",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=raw_data_s3,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="features",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/data/batch/features",
        )
    ],
    code="scripts/preprocess.py",
)

# ========================
# 6. Load trained model
# ========================
model = SKLearnModel(
    model_data=model_artifact_s3,
    role=role,
    entry_point="scripts/batch_inference.py",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)

transformer = Transformer(
    model_name=model.name,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=output_s3,
    sagemaker_session=session,
)

# ========================
# 7. Batch Inference Step
# ========================
batch_step = TransformStep(
    name="BatchInference",
    transformer=transformer,
    inputs=TransformInput(
        data=preprocess_step.properties
        .ProcessingOutputConfig.Outputs["features"]
        .S3Output.S3Uri,
        content_type="text/csv",
        split_type="Line",
    ),
)

# ========================
# 8. Pipeline Definition
# ========================
pipeline = Pipeline(
    name="TaxiBatchInferencePipeline",
    parameters=[raw_data_s3, model_artifact_s3, output_s3],
    steps=[preprocess_step, batch_step],
    sagemaker_session=session,
)
