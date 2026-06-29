import logging
from functools import lru_cache

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ordo_ai.config import get_settings
from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)

DELETE_TAGS = {"IP", "RP", "FS"}


@lru_cache
def _load():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.disfluency_model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        settings.disfluency_model_path
    )
    model.eval()
    return tokenizer, model


def predict_disfluency(text: str) -> dict:
    """Run normalized text through the fine-tuned disfluency tagger.

    Returns tokens (word-level), labels (BIO tag per word, first-subword
    label only), and spans (label/text/start/end, end exclusive, word indices).
    """
    tokenizer, model = _load()
    settings = get_settings()

    tokens = text.split()
    if not tokens:
        return {"tokens": [], "labels": [], "spans": []}

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=settings.max_seq_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids()

    with torch.no_grad():
        logits = model(**encoding).logits[0]
    pred_ids = logits.argmax(dim=-1).tolist()
    id2label = model.config.id2label

    word_labels = [None] * len(tokens)
    prev_word_id = None
    for tok_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id != prev_word_id:
            word_labels[word_id] = id2label[pred_ids[tok_idx]]
        prev_word_id = word_id
    word_labels = [lbl if lbl is not None else "O" for lbl in word_labels]

    spans = []
    i = 0
    while i < len(tokens):
        lbl = word_labels[i]
        if lbl.startswith("B-"):
            span_type = lbl[2:]
            j = i + 1
            while j < len(tokens) and word_labels[j] == f"I-{span_type}":
                j += 1
            spans.append(
                {
                    "label": span_type,
                    "text": " ".join(tokens[i:j]),
                    "start": i,
                    "end": j,
                }
            )
            i = j
        else:
            i += 1

    return {"tokens": tokens, "labels": word_labels, "spans": spans}


def repair(text: str) -> str:
    """Delete IP/RP/FS spans; for RC (repeat), drop the first occurrence and
    keep the second as the fluent surface form. RM (repair) is left untouched.
    """
    result = predict_disfluency(text)
    tokens, spans = result["tokens"], result["spans"]
    drop_indices = set()
    for span in spans:
        if span["label"] in DELETE_TAGS:
            drop_indices.update(range(span["start"], span["end"]))
        elif span["label"] == "RC":
            drop_indices.update(range(span["start"], span["end"] - 1))
    kept = [tok for idx, tok in enumerate(tokens) if idx not in drop_indices]
    return " ".join(kept)


def run(state: OrderState) -> OrderState:
    result = predict_disfluency(state["normalized_text"])
    repaired_text = repair(state["normalized_text"])
    logger.debug(
        "disfluency: tags=%r repaired_text=%r", result["labels"], repaired_text
    )
    return {
        "disfluency_tags": result["labels"],
        "repaired_text": repaired_text,
    }
