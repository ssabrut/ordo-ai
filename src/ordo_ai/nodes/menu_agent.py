import logging
from functools import lru_cache

from mlx_lm import generate, load

from ordo_ai.state.schemas import OrderState
from ordo_ai.tools.menu import find_menu_item, search_menu, search_menu_semantic

logger = logging.getLogger(__name__)

_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

_INQUIRY_SYSTEM = (
    "Kamu adalah asisten pemesanan makanan yang ramah dan membantu. "
    "Jawab pertanyaan pelanggan tentang menu dengan bahasa Indonesia yang natural dan hangat. "
    "Fokus pada deskripsi dan bahan-bahan menu. Jangan sebutkan harga. Jawab singkat (1-2 kalimat)."
)


@lru_cache
def _load_llm():
    logger.info("menu_agent: loading MLX model %s", _MODEL_ID)
    return load(_MODEL_ID)


def _dish_query(state: OrderState) -> str | None:
    for ent in state.get("entities", []):
        if ent["label"] in ("FOOD_ITEM", "DRINK_ITEM"):
            return ent["text"]
    return None


def _llm_inquiry_response(user_question: str, items: list[dict]) -> str:
    model, tokenizer = _load_llm()
    menu_info = "\n".join(
        f"- {item['name']}" + (f": {item['description']}" if item.get("description") else "")
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
    is_inquiry = state.get("intent") in ("menu_inquiry", "ask_recommendation")

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
        if len(results) == 1:
            result["last_discussed_item"] = results[0]
    else:
        lines = [
            f"{item['name']} - Rp{item['price']:,}".replace(",", ".") for item in results
        ]
        response = "Berikut menu yang tersedia: " + "; ".join(lines)
        result = {"agent_response": response}

    logger.debug("menu_agent: result=%r", result)
    return result
