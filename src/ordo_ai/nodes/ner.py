import logging
from functools import lru_cache

import torch
from transformers import AutoTokenizer

from ordo_ai.config import get_settings
from ordo_ai.models.ner_crf import BertCrfForTokenClassification
from ordo_ai.state.schemas import EntitySpan, OrderState

logger = logging.getLogger(__name__)


@lru_cache
def _load():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.ner_model_path)
    model = BertCrfForTokenClassification.from_pretrained(settings.ner_model_path)
    model.eval()
    return tokenizer, model


def predict_entities(text: str) -> dict:
    """Run text through the fine-tuned entity tagger.

    Returns tokens (word-level, lowercased), labels (BIO tag per word), and
    spans (label/text/start/end, end exclusive, word indices).
    """
    tokenizer, model = _load()
    settings = get_settings()

    tokens = text.lower().split()
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
        pred_ids = model.decode(encoding["input_ids"], encoding["attention_mask"])[0]
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


def run(state: OrderState) -> OrderState:
    result = predict_entities(state["repaired_text"])
    entities: list[EntitySpan] = [
        {"text": s["text"], "label": s["label"], "start": s["start"], "end": s["end"]}
        for s in result["spans"]
    ]
    logger.debug("ner: entities=%r", entities)
    return {"entities": entities}
