from difflib import SequenceMatcher

from ordo_ai.state.schemas import CartItem


def find_cart_index(cart: list[CartItem], name: str) -> int | None:
    for idx, item in enumerate(cart):
        if item["name"].lower() == name.lower():
            return idx
    return None


def _strip_nya(word: str) -> str:
    """Strip possessive/focus suffix -nya from Indonesian word."""
    return word[:-3] if word.endswith("nya") and len(word) > 4 else word


def find_cart_index_fuzzy(cart: list[CartItem], query: str, threshold: float = 0.55) -> int | None:
    """Fuzzy-match query against cart item names.

    Requires at least one query word (after stripping -nya suffix) to appear in
    the cart item name to avoid cross-word false positives (e.g. 'mie goreng'
    matching 'Ayam Goreng'). Boosts items whose name contains ALL query words.
    """
    raw_query_words = query.lower().split()
    # normalise: strip -nya suffix so "spesialnya" matches "spesial"
    query_words_norm = {_strip_nya(w) for w in raw_query_words}
    best_idx, best_score = None, 0.0
    for idx, item in enumerate(cart):
        item_words = set(item["name"].lower().split())
        # require every normalised query word to appear (as prefix) in some item word
        def _word_matches(qw: str) -> bool:
            return any(iw == qw or iw.startswith(qw) for iw in item_words)
        if not all(_word_matches(qw) for qw in query_words_norm):
            continue
        score = SequenceMatcher(None, query.lower(), item["name"].lower()).ratio()
        if query_words_norm.issubset(item_words):
            score += 0.2
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx if best_score >= threshold else None


def add_item(
    cart: list[CartItem], menu_item: dict, quantity: int = 1, notes: list[str] | None = None
) -> tuple[list[CartItem], str]:
    """Merge a menu item into the cart (qty += if already present), return updated cart + message."""
    cart = list(cart)
    idx = find_cart_index(cart, menu_item["name"])

    if idx is not None:
        cart[idx]["quantity"] += quantity
        message = f"{menu_item['name']} ditambah {quantity}, total {cart[idx]['quantity']}."
    else:
        cart.append(
            {
                "menu_id": menu_item["id"],
                "name": menu_item["name"],
                "price": menu_item["price"],
                "quantity": quantity,
                "notes": notes or [],
            }
        )
        message = f"{menu_item['name']} x{quantity} ditambahkan ke pesanan."

    return cart, message
