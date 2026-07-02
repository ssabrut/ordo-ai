from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer

from ordo_ai.config import get_settings

_model_path: str | None = None


def set_model_path(path: str) -> None:
    """Override the model path (e.g. after downloading from mlflow at startup)."""
    global _model_path
    _model_path = path
    _load.cache_clear()


@lru_cache
def _load():
    """Reuse the fine-tuned IndoBERT NER backbone as a sentence encoder."""
    path = _model_path or get_settings().ner_model_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.eval()
    return tokenizer, model


def embed_text(text: str) -> list[float]:
    """Mean-pool last hidden states (attention-masked) into a single vector."""
    tokenizer, model = _load()
    encoding = tokenizer(text, truncation=True, max_length=get_settings().max_seq_length, return_tensors="pt")

    with torch.no_grad():
        hidden = model(**encoding).last_hidden_state[0]

    mask = encoding["attention_mask"][0].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1)
    normalized = pooled / pooled.norm().clamp(min=1e-9)
    return normalized.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]
