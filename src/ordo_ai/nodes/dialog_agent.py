from ordo_ai.state.schemas import OrderState

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

    if intent == "deny":
        return {"cart": [], "agent_response": response}
    if intent == "repeat_request":
        return {"agent_response": f"{response} {_format_cart(state.get('cart', []))}"}
    return {"agent_response": response}
