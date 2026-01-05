import os
from mlproject.pipelines.batch_inference_pipeline import pipeline

pipeline.upsert(role_arn=os.environ["SAGEMAKER_ROLE_ARN"])

pipeline.start(
    parameters={
        "RawDataS3": "s3://your-bucket/data/raw",
        "ModelArtifactS3": "s3://your-bucket/models/model.tar.gz",
        "PredictionsS3": "s3://your-bucket/predictions/run-001",
    }
)
