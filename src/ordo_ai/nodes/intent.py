from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ordo_ai.config import get_settings
from ordo_ai.state.schemas import OrderState


@lru_cache
def _load():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.intent_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(settings.intent_model_path)
    model.eval()
    return tokenizer, model


def predict_intent(text: str) -> dict:
    tokenizer, model = _load()
    settings = get_settings()

    encoding = tokenizer(text, truncation=True, max_length=settings.max_seq_length, return_tensors="pt")

    with torch.no_grad():
        logits = model(**encoding).logits[0]
    probs = F.softmax(logits, dim=-1)
    pred_id = probs.argmax().item()
    id2label = model.config.id2label

    return {
        "intent": id2label[pred_id],
        "confidence": probs[pred_id].item(),
        "probs": {id2label[i]: p.item() for i, p in enumerate(probs)},
    }


def run(state: OrderState) -> OrderState:
    result = predict_intent(state["repaired_text"])
    return {
        "intent": result["intent"],
        "intent_confidence": result["confidence"],
        "intent_probs": result["probs"],
    }
