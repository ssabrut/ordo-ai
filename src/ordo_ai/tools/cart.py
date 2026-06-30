from ordo_ai.state.schemas import CartItem


def find_cart_index(cart: list[CartItem], name: str) -> int | None:
    for idx, item in enumerate(cart):
        if item["name"].lower() == name.lower():
            return idx
    return None


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
