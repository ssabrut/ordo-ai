from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.menu import search_menu


def _dish_query(state: OrderState) -> str | None:
    for ent in state.get("entities", []):
        if ent["label"] in ("DISH", "DRINK"):
            return ent["text"]
    return None


def run(state: OrderState) -> OrderState:
    query = _dish_query(state)
    results = search_menu(query=query)

    if not results:
        return {"agent_response": "Maaf, menu yang Anda maksud tidak ditemukan."}

    lines = [
        f"{item['name']} - Rp{item['price']:,}".replace(",", ".") for item in results
    ]
    return {"agent_response": "Berikut menu yang tersedia: " + "; ".join(lines)}
