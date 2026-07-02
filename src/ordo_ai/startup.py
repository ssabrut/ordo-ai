"""Model loading at process startup, not at first inference / graph compile.

Loads disfluency + NER + intent model weights off local disk before the
server starts serving requests.
"""

import logging

from ordo_ai.config import get_settings
from ordo_ai.nodes import disfluency, intent, menu_agent, ner
from ordo_ai.tools import embeddings

logger = logging.getLogger(__name__)


def load_models() -> None:
    settings = get_settings()

    disfluency.set_model_path(settings.disfluency_model_path)
    ner.set_model_path(settings.ner_model_path)
    embeddings.set_model_path(settings.ner_model_path)
    intent.set_model_path(settings.intent_model_path)

    logger.info("startup: loading model weights into memory")
    disfluency._load()
    ner._load()
    intent._load()
    embeddings._load()
    menu_agent._load_llm()
    logger.info("startup: all models loaded")
