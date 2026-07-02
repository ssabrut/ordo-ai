"""Model loading at process startup, not at first inference / graph compile.

Downloads the latest disfluency + NER model artifacts from mlflow and forces
weights off disk before the server starts serving requests.
"""

import logging
import tempfile
from pathlib import Path

from ordo_ai.config import get_settings
from ordo_ai.nodes import disfluency, intent, menu_agent, ner
from ordo_ai.tools import embeddings
from ordo_ai.tracking import download_latest_model

logger = logging.getLogger(__name__)


def load_models() -> None:
    settings = get_settings()
    cache_dir = Path(tempfile.gettempdir()) / "ordo_ai_mlflow_models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("startup: downloading disfluency model from mlflow experiment %r", settings.mlflow_disfluency_experiment)
    disfluency_path = download_latest_model(
        settings.mlflow_disfluency_experiment, str(cache_dir / "disfluency")
    )
    disfluency.set_model_path(disfluency_path)

    logger.info("startup: downloading NER model from mlflow experiment %r", settings.mlflow_ner_experiment)
    ner_path = download_latest_model(
        settings.mlflow_ner_experiment, str(cache_dir / "ner")
    )
    ner.set_model_path(ner_path)
    embeddings.set_model_path(ner_path)

    logger.info("startup: downloading intent model from mlflow experiment %r", settings.mlflow_intent_experiment)
    intent_path = download_latest_model(
        settings.mlflow_intent_experiment, str(cache_dir / "intent")
    )
    intent.set_model_path(intent_path)

    logger.info("startup: loading model weights into memory")
    disfluency._load()
    ner._load()
    intent._load()
    embeddings._load()
    menu_agent._load_llm()
    logger.info("startup: all models loaded")
