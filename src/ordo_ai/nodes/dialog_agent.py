import logging
import re

from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.cart import add_item, find_cart_index

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

    # confirm — resolve pending_item only when actively awaiting disambiguation
    pending = state.get("pending_item")
    if state.get("needs_clarification") and pending and pending.get("candidates"):
        candidates = pending["candidates"]
        repaired = state.get("repaired_text", "")
        pick_idx = _pick_candidate_index(repaired)
        # also try matching candidate name directly from repaired text
        if pick_idx is None:
            lower = repaired.lower()
            for i, c in enumerate(candidates):
                if c["name"].lower() in lower or lower in c["name"].lower():
                    pick_idx = i
                    break
        if pick_idx is not None and 0 <= pick_idx < len(candidates):
            menu_item = candidates[pick_idx]
            cart = list(state.get("cart", []))

            # if this was a swap, remove the original item first
            remove_name = pending.get("remove_name")
            if remove_name:
                idx = find_cart_index(cart, remove_name)
                if idx is not None:
                    cart.pop(idx)

            cart, message = add_item(cart, menu_item, pending["quantity"], pending["notes"])
            result = {
                "cart": cart,
                "pending_item": None,
                "needs_clarification": False,
                "agent_response": message,
            }
            logger.debug("dialog_agent: resolved pending_item -> %r (removed %r)", menu_item["name"], remove_name)
            return result

        # user confirmed but didn't pick a number — re-ask
        options = ", ".join(
            f"{i+1}. {c['name']}" for i, c in enumerate(candidates)
        )
        msg = f"Pilih nomor: {options}."
        result = {
            "needs_clarification": True,
            "clarification_message": msg,
            "agent_response": msg,
        }
        logger.debug("dialog_agent: re-asking clarification")
        return result

    result = {"agent_response": _RESPONSES["confirm"]}
    logger.debug("dialog_agent: result=%r", result)
    return result
