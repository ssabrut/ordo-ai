import logging

from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.menu import search_menu

logger = logging.getLogger(__name__)


def _dish_query(state: OrderState) -> str | None:
    for ent in state.get("entities", []):
        if ent["label"] in ("DISH", "DRINK"):
            return ent["text"]
    return None


def run(state: OrderState) -> OrderState:
    query = _dish_query(state)
    results = search_menu(query=query)
    logger.debug("menu_agent: query=%r results=%r", query, [r["name"] for r in results])

    if not results:
        result = {"agent_response": "Maaf, menu yang Anda maksud tidak ditemukan."}
        logger.debug("menu_agent: result=%r", result)
        return result

    lines = [
        f"{item['name']} - Rp{item['price']:,}".replace(",", ".") for item in results
    ]
    result = {"agent_response": "Berikut menu yang tersedia: " + "; ".join(lines)}
    logger.debug("menu_agent: result=%r", result)
    return result
