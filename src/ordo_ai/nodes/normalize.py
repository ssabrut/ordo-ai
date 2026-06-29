import logging
import re

from num2words import num2words

from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)

_PUNCT_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NUMBER_PATTERN = re.compile(r"\d+")


def remove_punctuation(text: str) -> str:
    return _PUNCT_PATTERN.sub(" ", text)


def normalize_numbers(text: str, lang: str = "id") -> str:
    return _NUMBER_PATTERN.sub(lambda m: num2words(int(m.group()), lang=lang), text)


def normalize(text: str) -> str:
    text = remove_punctuation(text)
    text = normalize_numbers(text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip().lower()
    return text


def run(state: OrderState) -> OrderState:
    result = normalize(state["raw_text"])
    logger.debug("normalize: normalized_text=%r", result)
    return {"normalized_text": result}
