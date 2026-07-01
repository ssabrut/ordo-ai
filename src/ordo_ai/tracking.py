import os

import mlflow

from ordo_ai.config import get_settings


def init_experiment(name: str) -> None:
    settings = get_settings()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(name)
