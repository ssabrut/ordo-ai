import logging

from mlx_lm import generate

from ordo_ai.nodes.intent import _load as _load_llm
from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.cart import add_item
from ordo_ai.tools.menu import find_menu_item, search_menu, search_menu_semantic

logger = logging.getLogger(__name__)

_INQUIRY_SYSTEM = (
    "Kamu adalah asisten pemesanan makanan yang ramah dan membantu. "
    "Jawab pertanyaan pelanggan tentang menu dengan bahasa Indonesia yang natural dan hangat. "
    "Sertakan nama menu, harga, dan deskripsi. Jawab singkat (1-2 kalimat)."
)


def _dish_query(state: OrderState) -> str | None:
    for ent in state.get("entities", []):
        if ent["label"] in ("DISH", "DRINK"):
            return ent["text"]
    return None


def _llm_inquiry_response(user_question: str, items: list[dict]) -> str:
    model, tokenizer = _load_llm()
    menu_info = "\n".join(
        f"- {item['name']}: Rp{item['price']:,}".replace(",", ".") +
        (f" — {item['description']}" if item.get("description") else "")
        for item in items
    )
    user_msg = f"Pertanyaan pelanggan: {user_question}\nInfo menu:\n{menu_info}"
    messages = [
        {"role": "system", "content": _INQUIRY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False).strip()


def run(state: OrderState) -> OrderState:
    query = _dish_query(state)
    is_inquiry = state.get("intent") == "menu_inquiry"

    if query:
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

    if is_inquiry:
        user_question = state.get("normalized_text", query or "")
        response = _llm_inquiry_response(user_question, results)
        result = {"agent_response": response}
    else:
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
