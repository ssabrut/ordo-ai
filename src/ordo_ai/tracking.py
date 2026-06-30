import mlflow

from ordo_ai.config import get_settings


def init_experiment(name: str) -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(name)
