# src/mlproject/pipelines/inference_pipeline.py

# ========================
# 1. Imports
# ========================
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

# ========================
# 2. AWS Session & Config
# ========================
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name()

# ========================
# 3. ScriptProcessor
# (used for preprocess + feature engineering)
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
    name="PreprocessBatch",
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
            destination=f"s3://{bucket}/data/batch/clean",
        )
    ],
    code="../scripts/preprocess.py",
    job_arguments=[
        "--test_path", "/opt/ml/processing/input/test.csv",
        "--output_test", "/opt/ml/processing/output/test.csv",
    ],
)

# ========================
# 5. Feature Engineering Step
# ========================
feature_step = ProcessingStep(
    name="FeatureEngineeringBatch",
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
            source="/opt/ml/processing/output/features",
            destination=f"s3://{bucket}/data/batch/features",
        ),
    ],
    code="../scripts/feature_engineering.py",
    job_arguments=[
        "--test_input", "/opt/ml/processing/input/test.csv",
        "--eval_dir", "/opt/ml/processing/output/features",
    ],
)

# ========================
# 6. Load the trained model
# ========================
model = Model(
    image_uri=processor.image_uri,
    model_data=f"s3://{bucket}/models/taxi_model/model.tar.gz",
    role=role,
)

# ========================
# 7. Batch Transform Step
# ========================
transformer = Transformer(
    model_name="TaxiBatchModel",
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{bucket}/outputs/predictions/",
)

batch_step = TransformStep(
    name="BatchInference",
    transformer=transformer,
    inputs=feature_step.properties
        .ProcessingOutputConfig.Outputs["features"]
        .S3Output.S3Uri,
)

# ========================
# 8. Pipeline Definition
# ========================
pipeline = Pipeline(
    name="TaxiBatchInferencePipeline",
    steps=[
        preprocess_step,
        feature_step,
        batch_step,
    ],
    sagemaker_session=session,
)

# ========================
# 9. Run the Pipeline
# ========================
pipeline.upsert(role_arn=role)
pipeline.start()

print("✅ Batch inference pipeline started successfully!")
