import os
from mlproject.pipelines.training_pipeline import pipeline

pipeline.upsert(role_arn=os.environ["SAGEMAKER_ROLE_ARN"])

pipeline.start(
    parameters={
        "TrainDataS3": "s3://your-bucket/data/train",
        "TargetColumn": "trip_duration",
    }
)
