import logging

from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.cart import add_item
from ordo_ai.tools.menu import find_menu_item, search_menu, search_menu_semantic

logger = logging.getLogger(__name__)


def _dish_query(state: OrderState) -> str | None:
    for ent in state.get("entities", []):
        if ent["label"] in ("DISH", "DRINK"):
            return ent["text"]
    return None


def run(state: OrderState) -> OrderState:
    query = _dish_query(state)

    if query:
        # NER caught a specific dish/drink span -> resolve to one confident item.
        menu_item = find_menu_item(query)
        results = [menu_item] if menu_item else search_menu(query=query)
        logger.debug("menu_agent: fuzzy query=%r results=%r", query, [r["name"] for r in results])
    else:
        free_text = state.get("repaired_text", "")
        menu_item = None
        results = search_menu_semantic(free_text) if free_text else []
        logger.debug("menu_agent: semantic query=%r results=%r", free_text, [r["name"] for r in results])

    if not results:
        result = {"agent_response": "Maaf, menu yang Anda maksud tidak ditemukan."}
        logger.debug("menu_agent: result=%r", result)
        return result

    lines = [
        f"{item['name']} - Rp{item['price']:,}".replace(",", ".") for item in results
    ]
    response = "Berikut menu yang tersedia: " + "; ".join(lines)

    if menu_item:
        cart, message = add_item(state.get("cart", []), menu_item)
        result = {"cart": cart, "agent_response": f"{response} {message}"}
    else:
        result = {"agent_response": response}

    logger.debug("menu_agent: result=%r", result)
    return result
