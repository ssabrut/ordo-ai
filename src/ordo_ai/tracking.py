import os

import mlflow
from mlflow.tracking import MlflowClient

from ordo_ai.config import get_settings


def _configure() -> None:
    settings = get_settings()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def init_experiment(name: str) -> None:
    _configure()
    mlflow.set_experiment(name)


def download_latest_model(experiment_name: str, dest_dir: str) -> str:
    """Download the 'model' artifact dir from the most recent finished run in
    an experiment. Used at process startup to fetch model weights from mlflow
    instead of reading a checked-in local path.
    """
    _configure()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"mlflow experiment not found: {experiment_name!r}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"no finished runs in mlflow experiment: {experiment_name!r}")

    run_id = runs[0].info.run_id
    return client.download_artifacts(run_id, "model", dest_dir)
