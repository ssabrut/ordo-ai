import logging

from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)

_RESPONSES = {
    "confirm": "Baik, pesanan dikonfirmasi.",
    "deny": "Baik, pesanan dibatalkan.",
    "repeat_request": "Tentu, saya ulangi pesanan Anda.",
}


def _format_cart(cart: list[dict]) -> str:
    if not cart:
        return "Pesanan Anda saat ini kosong."
    lines = [f"{item['quantity']}x {item['name']}" for item in cart]
    return "Pesanan Anda: " + ", ".join(lines) + "."


def run(state: OrderState) -> OrderState:
    intent = state["intent"]
    response = _RESPONSES[intent]
    logger.debug("dialog_agent: intent=%r", intent)

    if intent == "deny":
        result = {"cart": [], "agent_response": response}
    elif intent == "repeat_request":
        result = {"agent_response": f"{response} {_format_cart(state.get('cart', []))}"}
    else:
        result = {"agent_response": response}

    logger.debug("dialog_agent: result=%r", result)
    return result
