import json
import logging
import re
from functools import lru_cache

from mlx_lm import generate, load

from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)

_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

_VALID_LABELS = [
    "chitchat_oos",
    "confirm",
    "deny",
    "menu_inquiry",
    "order_add_item",
    "order_cancel",
    "order_create",
    "order_modify_quantity",
    "order_remove_item",
    "order_swap",
    "repeat_request",
]

_LABEL_DESCRIPTIONS = {
    "order_create": "user wants to start a new order",
    "order_add_item": "user wants to add a food/drink item to their order",
    "order_remove_item": "user wants to remove a specific item from their order",
    "order_cancel": "user wants to cancel the entire order",
    "order_modify_quantity": "user wants to change the quantity of an item already in their order",
    "order_swap": "user wants to replace/swap one item in their order with another (e.g. 'ganti X ke Y', 'tukar X dengan Y')",
    "menu_inquiry": "user asks about available menu items, prices, or descriptions",
    "repeat_request": "user asks to recap, repeat, or review what is currently in their order/cart",
    "confirm": "user confirms or agrees (yes, oke, betul, lanjut, dll)",
    "deny": "user denies or disagrees (tidak, bukan, gak jadi, dll)",
    "chitchat_oos": "out-of-scope or general chit-chat unrelated to ordering food",
}

_LABEL_BLOCK = "\n".join(f'  "{k}": {v}' for k, v in _LABEL_DESCRIPTIONS.items())

_SYSTEM_PROMPT = (
    "You are an intent classifier for an Indonesian food-ordering voice assistant.\n"
    "Classify the user utterance into exactly one of these intents:\n"
    + _LABEL_BLOCK + "\n\n"
    "Output ONLY a single JSON object with two fields:\n"
    '  "intent": one of the intent names above\n'
    '  "confidence": float 0.0–1.0\n'
    "No explanation. No markdown. No extra text.\n\n"
    "Example output:\n"
    '{"intent": "menu_inquiry", "confidence": 0.95}'
)


def _regex_rescue(raw: str) -> tuple[str | None, float]:
    """Extract intent label and confidence from malformed model output via regex."""
    # find first valid label name appearing in the raw string
    pattern = "|".join(re.escape(label) for label in _VALID_LABELS)
    match = re.search(pattern, raw)
    if not match:
        return None, 0.0
    intent = match.group(0)
    # find first float/int in the string (confidence)
    num_match = re.search(r'0?\.\d+|\b1\.0\b|\b[01]\b', raw)
    confidence = float(num_match.group(0)) if num_match else 0.9
    return intent, min(max(confidence, 0.0), 1.0)


@lru_cache
def _load():
    logger.info("intent: loading MLX model %s", _MODEL_ID)
    model, tokenizer = load(_MODEL_ID)
    return model, tokenizer


def predict_intent(text: str) -> dict:
    model, tokenizer = _load()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    raw = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)

    try:
        cleaned = raw.strip()
        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned
        if not cleaned.endswith("}"):
            cleaned = cleaned + "}"
        parsed = json.loads(cleaned)

        if "intent" in parsed:
            intent = parsed["intent"]
            confidence = float(parsed["confidence"])
        else:
            # model returned {"label": score} shorthand
            label_keys = [k for k in parsed if k in _VALID_LABELS]
            if not label_keys:
                raise ValueError(f"no valid label key in {parsed}")
            intent = max(label_keys, key=lambda k: float(parsed[k]))
            confidence = float(parsed[intent])

        if intent not in _VALID_LABELS:
            raise ValueError(f"unknown intent label: {intent!r}")
    except Exception as exc:
        logger.warning("intent: parse failed (%s), raw=%r, trying regex rescue", exc, raw)
        intent, confidence = _regex_rescue(raw)
        if intent is None:
            logger.warning("intent: regex rescue failed, defaulting to chitchat_oos")
            intent = "chitchat_oos"
            confidence = 0.0

    return {
        "intent": intent,
        "confidence": confidence,
        "probs": {label: (confidence if label == intent else 0.0) for label in _VALID_LABELS},
    }


def run(state: OrderState) -> OrderState:
    result = predict_intent(state["normalized_text"])
    logger.debug(
        "intent: intent=%r confidence=%.4f",
        result["intent"],
        result["confidence"],
    )
    return {
        "intent": result["intent"],
        "intent_confidence": result["confidence"],
        "intent_probs": result["probs"],
    }
