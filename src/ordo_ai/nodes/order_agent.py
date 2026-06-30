import logging

from ordo_ai.state.schemas import CartItem, EntitySpan, OrderState
from ordo_ai.tools.cart import add_item, find_cart_index
from ordo_ai.tools.menu import find_menu_item

logger = logging.getLogger(__name__)

_NUMBER_WORDS = {
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
    "sepuluh": 10,
}


def _parse_quantity(text: str) -> int:
    text = text.lower().strip()
    if text.isdigit():
        return int(text)
    return _NUMBER_WORDS.get(text, 1)


def _group_entities(entities: list[EntitySpan]) -> list[dict]:
    """Attach each DISH/DRINK/REMOVE span to its nearest QUANTITY/MODIFIER spans (qty
    can precede, e.g. "dua nasi goreng", or follow, e.g. "nasi goreng dua porsi").
    """
    spans = sorted(entities, key=lambda e: e["start"])
    anchors = [
        i for i, e in enumerate(spans) if e["label"] in ("DISH", "DRINK", "REMOVE")
    ]
    if not anchors:
        return []

    items = [
        {
            "name": spans[i]["text"],
            "label": spans[i]["label"],
            "quantity": 1,
            "notes": [],
        }
        for i in anchors
    ]

    for i, ent in enumerate(spans):
        if ent["label"] not in ("QUANTITY", "MODIFIER", "ADD_ON", "SIZE"):
            continue
        nearest = min(range(len(anchors)), key=lambda k: abs(anchors[k] - i))
        if ent["label"] == "QUANTITY":
            items[nearest]["quantity"] = _parse_quantity(ent["text"])
        else:
            items[nearest]["notes"].append(ent["text"])

    return items


def run(state: OrderState) -> OrderState:
    intent = state["intent"]
    cart = list(state.get("cart", []))
    parsed_items = _group_entities(state.get("entities", []))
    logger.debug("order_agent: intent=%r parsed_items=%r", intent, parsed_items)

    if intent == "order_cancel":
        cart = []
        result = {"cart": cart, "agent_response": "Pesanan dibatalkan."}
        logger.debug("order_agent: result=%r", result)
        return result

    if not parsed_items:
        result = {
            "cart": cart,
            "agent_response": "Maaf, saya tidak menangkap item pesanan Anda.",
        }
        logger.debug("order_agent: result=%r", result)
        return result

    responses = []
    for parsed in parsed_items:
        menu_item = find_menu_item(parsed["name"])

        if intent == "order_remove_item" or parsed["label"] == "REMOVE":
            if menu_item:
                idx = find_cart_index(cart, menu_item["name"])
                if idx is not None:
                    cart.pop(idx)
                    responses.append(f"{menu_item['name']} dihapus dari pesanan.")
                    continue
            responses.append(f"{parsed['name']} tidak ditemukan di pesanan.")
            continue

        if not menu_item:
            responses.append(f"Maaf, menu '{parsed['name']}' tidak tersedia.")
            continue

        idx = find_cart_index(cart, menu_item["name"])
        if intent == "order_modify_quantity" and idx is not None:
            cart[idx]["quantity"] = parsed["quantity"]
            responses.append(
                f"Jumlah {menu_item['name']} diubah menjadi {parsed['quantity']}."
            )
        else:
            cart, message = add_item(cart, menu_item, parsed["quantity"], parsed["notes"])
            responses.append(message)

    result = {"cart": cart, "agent_response": " ".join(responses)}
    logger.debug("order_agent: result=%r", result)
    return result
