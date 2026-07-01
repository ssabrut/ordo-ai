import logging
import re

from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.cart import add_item

logger = logging.getLogger(__name__)

_RESPONSES = {
    "confirm": "Baik, pesanan dikonfirmasi.",
    "deny": "Baik, pesanan dibatalkan.",
    "repeat_request": "Tentu, saya ulangi pesanan Anda.",
}

_ORDINAL_MAP = {
    "pertama": 1, "satu": 1, "1": 1,
    "kedua": 2, "dua": 2, "2": 2,
    "ketiga": 3, "tiga": 3, "3": 3,
    "keempat": 4, "empat": 4, "4": 4,
}


def _pick_candidate_index(text: str) -> int | None:
    text = text.lower().strip()
    for word, idx in _ORDINAL_MAP.items():
        if word in text:
            return idx - 1
    m = re.search(r'\b(\d+)\b', text)
    if m:
        return int(m.group(1)) - 1
    return None


def _format_cart(cart: list[dict]) -> str:
    if not cart:
        return "Pesanan Anda saat ini kosong."
    lines = [f"{item['quantity']}x {item['name']}" for item in cart]
    return "Pesanan Anda: " + ", ".join(lines) + "."


def run(state: OrderState) -> OrderState:
    intent = state["intent"]
    logger.debug("dialog_agent: intent=%r", intent)

    if intent == "deny":
        result = {"cart": [], "pending_item": None, "needs_clarification": False, "agent_response": _RESPONSES["deny"]}
        logger.debug("dialog_agent: result=%r", result)
        return result

    if intent == "repeat_request":
        result = {"agent_response": f"{_RESPONSES['repeat_request']} {_format_cart(state.get('cart', []))}"}
        logger.debug("dialog_agent: result=%r", result)
        return result

    # confirm — resolve pending_item if present
    pending = state.get("pending_item")
    if pending and pending.get("candidates"):
        candidates = pending["candidates"]
        pick_idx = _pick_candidate_index(state.get("repaired_text", ""))
        if pick_idx is not None and 0 <= pick_idx < len(candidates):
            menu_item = candidates[pick_idx]
            cart, message = add_item(
                list(state.get("cart", [])),
                menu_item,
                pending["quantity"],
                pending["notes"],
            )
            result = {
                "cart": cart,
                "pending_item": None,
                "needs_clarification": False,
                "agent_response": message,
            }
            logger.debug("dialog_agent: resolved pending_item -> %r", menu_item["name"])
            return result

        # user confirmed but didn't pick a number — re-ask
        options = ", ".join(
            f"{i+1}. {c['name']}" for i, c in enumerate(candidates)
        )
        result = {
            "needs_clarification": True,
            "agent_response": f"Pilih nomor: {options}.",
        }
        logger.debug("dialog_agent: re-asking clarification")
        return result

    result = {"agent_response": _RESPONSES["confirm"]}
    logger.debug("dialog_agent: result=%r", result)
    return result
