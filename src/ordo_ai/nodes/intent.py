import logging
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ordo_ai.config import get_settings
from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)

_model_path: str | None = None


def set_model_path(path: str) -> None:
    """Override the model path (e.g. after downloading from mlflow at startup)."""
    global _model_path
    _model_path = path
    _load.cache_clear()


@lru_cache
def _load():
    path = _model_path or get_settings().intent_model_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict_intent(text: str) -> dict:
    """Multi-label intent classification: returns every label whose sigmoid
    probability clears intent_confidence_threshold. Falls back to the single
    highest-probability label if none clear the threshold.
    """
    tokenizer, model = _load()
    settings = get_settings()

    encoding = tokenizer(
        text, truncation=True, max_length=settings.max_seq_length, return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**encoding).logits[0]
    probs = torch.sigmoid(logits)
    id2label = model.config.id2label

    all_probs = {id2label[i]: probs[i].item() for i in range(len(probs))}
    active = {
        label: p for label, p in all_probs.items() if p >= settings.intent_confidence_threshold
    }
    if not active:
        top_label = max(all_probs, key=all_probs.get)
        active = {top_label: all_probs[top_label]}

    return {"intents": list(active.keys()), "confidences": active, "probs": all_probs}


def run(state: OrderState) -> OrderState:
    result = predict_intent(state["normalized_text"])
    logger.debug(
        "intent: intents=%r confidences=%r", result["intents"], result["confidences"]
    )
    return {
        "intents": result["intents"],
        "intent_confidences": result["confidences"],
        "intent_probs": result["probs"],
    }
