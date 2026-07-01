import logging

from ordo_ai.state.schemas import CartItem, EntitySpan, OrderState, PendingItem
from ordo_ai.tools.cart import add_item, find_cart_index, find_cart_index_fuzzy
from ordo_ai.tools.menu import find_menu_item, find_menu_items

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


def _resolve_pending(state: OrderState) -> OrderState | None:
    """If state has an active disambiguation, try to resolve it from entities or repaired text."""
    pending = state.get("pending_item")
    if not (state.get("needs_clarification") and pending and pending.get("candidates")):
        return None

    candidates = pending["candidates"]
    cart = list(state.get("cart", []))

    # try entity match first
    for ent in state.get("entities", []):
        if ent["label"] in ("DISH", "DRINK"):
            lower = ent["text"].lower()
            for i, c in enumerate(candidates):
                if lower in c["name"].lower() or c["name"].lower() in lower:
                    remove_name = pending.get("remove_name")
                    if remove_name:
                        idx = find_cart_index(cart, remove_name)
                        if idx is not None:
                            cart.pop(idx)
                    cart, message = add_item(cart, c, pending["quantity"], pending["notes"])
                    logger.debug("order_agent: resolved pending via entity -> %r", c["name"])
                    return {"cart": cart, "pending_item": None, "needs_clarification": False, "agent_response": message}

    return None


def _consume_last_discussed(state: OrderState, cart: list, responses: list) -> dict | None:
    """If last_discussed_item is set and repaired text contains a reference to 'it' (itu/ini),
    add it to cart with the nearest orphaned QUANTITY, return updated cart."""
    last_item = state.get("last_discussed_item")
    if not last_item:
        return None
    repaired = state.get("repaired_text", "").lower()
    if not any(ref in repaired for ref in ("itu", "ini", "boleh", "oke", "mau itu")):
        return None

    # find first QUANTITY entity not preceded by a DISH entity
    spans = sorted(state.get("entities", []), key=lambda e: e["start"])
    qty = 1
    notes = []
    for ent in spans:
        if ent["label"] in ("DISH", "DRINK"):
            break
        if ent["label"] == "QUANTITY":
            qty = _parse_quantity(ent["text"])
        elif ent["label"] in ("MODIFIER", "ADD_ON", "SIZE"):
            notes.append(ent["text"])

    cart, message = add_item(cart, last_item, qty, notes)
    responses.append(message)
    logger.debug("order_agent: consumed last_discussed_item=%r x%d", last_item["name"], qty)
    return {"last_discussed_item": None}


def run(state: OrderState) -> OrderState:
    # resolve pending disambiguation before processing new intent
    resolved = _resolve_pending(state)
    if resolved is not None:
        return resolved

    intent = state["intent"]
    cart = list(state.get("cart", []))
    parsed_items = _group_entities(state.get("entities", []))
    logger.debug("order_agent: intent=%r parsed_items=%r", intent, parsed_items)

    # consume last_discussed_item if user referred to it ("itu", "ini", etc.)
    extra = {}
    responses = []
    consumed = _consume_last_discussed(state, cart, responses)
    if consumed is not None:
        extra.update(consumed)

    if intent == "order_cancel":
        cart = []
        result = {"cart": cart, "agent_response": "Pesanan dibatalkan."}
        logger.debug("order_agent: result=%r", result)
        return result

    if intent == "order_swap":
        if len(parsed_items) < 2:
            result = {"cart": cart, "agent_response": "Maaf, sebutkan item yang ingin diganti dan penggantinya."}
            logger.debug("order_agent: swap missing items")
            return result

        remove_parsed, add_parsed = parsed_items[0], parsed_items[1]

        # remove the first item from cart
        responses = []
        remove_name_resolved = None
        idx = find_cart_index_fuzzy(cart, remove_parsed["name"])
        if idx is not None:
            remove_name_resolved = cart[idx]["name"]
            cart.pop(idx)
            responses.append(f"{remove_name_resolved} dihapus dari pesanan.")
        else:
            responses.append(f"Menu '{remove_parsed['name']}' tidak ada di pesanan.")

        # add the second item (with ambiguity handling)
        candidates = find_menu_items(add_parsed["name"])
        if not candidates:
            responses.append(f"Maaf, menu '{add_parsed['name']}' tidak tersedia.")
            result = {"cart": cart, "agent_response": " ".join(responses)}
            return result

        if len(candidates) > 1:
            options = ", ".join(
                f"{i+1}. {c['name']} (Rp{c['price']:,})".replace(",", ".")
                for i, c in enumerate(candidates)
            )
            pending: PendingItem = {
                "name": add_parsed["name"],
                "quantity": add_parsed["quantity"],
                "notes": add_parsed["notes"],
                "candidates": candidates,
                "remove_name": remove_name_resolved,
            }
            msg = f"{''.join(responses)} Ada beberapa pilihan untuk '{add_parsed['name']}': {options}. Mau yang mana?"
            result = {
                "cart": cart,
                "pending_item": pending,
                "needs_clarification": True,
                "clarification_message": msg,
                "agent_response": msg,
            }
            logger.debug("order_agent: swap ambiguous add=%r", add_parsed["name"])
            return result

        cart, message = add_item(cart, candidates[0], add_parsed["quantity"], add_parsed["notes"])
        responses.append(message)
        result = {"cart": cart, "pending_item": None, "needs_clarification": False, "agent_response": " ".join(responses)}
        logger.debug("order_agent: swap result=%r", result)
        return result

    if not parsed_items:
        if responses:
            result = {**extra, "cart": cart, "last_discussed_item": None, "agent_response": " ".join(responses)}
        else:
            result = {"cart": cart, "agent_response": "Maaf, saya tidak menangkap item pesanan Anda."}
        logger.debug("order_agent: result=%r", result)
        return result

    for parsed in parsed_items:
        if intent == "order_remove_item" or parsed["label"] == "REMOVE":
            idx = find_cart_index_fuzzy(cart, parsed["name"])
            if idx is not None:
                removed_name = cart[idx]["name"]
                cart.pop(idx)
                responses.append(f"{removed_name} dihapus dari pesanan.")
            else:
                responses.append(f"{parsed['name']} tidak ditemukan di pesanan.")
            continue

        candidates = find_menu_items(parsed["name"])

        if not candidates:
            responses.append(f"Maaf, menu '{parsed['name']}' tidak tersedia.")
            continue

        if len(candidates) > 1:
            options = ", ".join(
                f"{i+1}. {c['name']} (Rp{c['price']:,})".replace(",", ".")
                for i, c in enumerate(candidates)
            )
            pending: PendingItem = {
                "name": parsed["name"],
                "quantity": parsed["quantity"],
                "notes": parsed["notes"],
                "candidates": candidates,
            }
            result = {
                "cart": cart,
                "pending_item": pending,
                "needs_clarification": True,
                "clarification_message": f"Ada beberapa pilihan untuk '{parsed['name']}': {options}. Mau yang mana?",
                "agent_response": f"Ada beberapa pilihan untuk '{parsed['name']}': {options}. Mau yang mana?",
            }
            logger.debug("order_agent: ambiguous item=%r candidates=%r", parsed["name"], [c["name"] for c in candidates])
            return result

        menu_item = candidates[0]
        idx = find_cart_index(cart, menu_item["name"])
        if intent == "order_modify_quantity" and idx is not None:
            cart[idx]["quantity"] = parsed["quantity"]
            responses.append(
                f"Jumlah {menu_item['name']} diubah menjadi {parsed['quantity']}."
            )
        else:
            cart, message = add_item(cart, menu_item, parsed["quantity"], parsed["notes"])
            responses.append(message)

    result = {**extra, "cart": cart, "agent_response": " ".join(responses)}
    if extra:
        result["last_discussed_item"] = None
    logger.debug("order_agent: result=%r", result)
    return result
