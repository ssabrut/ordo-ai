import json
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import chromadb

from ordo_ai.config import get_settings
from ordo_ai.tools.embeddings import embed_text, embed_texts

_MENU_PATH = Path(__file__).resolve().parents[3] / "data" / "menu.json"


class MenuItem(dict):
    pass


@lru_cache
def load_menu() -> list[dict]:
    with open(_MENU_PATH, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for category in data["categories"]:
        items.extend(category["menus"])
    return items


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_menu_item(name: str, threshold: float = 0.55) -> dict | None:
    """Fuzzy-match an entity span (e.g. NER DISH/DRINK text) against the menu catalog."""
    items = load_menu()
    best, best_score = None, 0.0
    for item in items:
        score = _similarity(name, item["name"])
        if score > best_score:
            best, best_score = item, score
    if best is not None and best_score >= threshold:
        return best
    return None


def search_menu(query: str | None = None, category: str | None = None) -> list[dict]:
    items = load_menu()
    if category:
        items = [i for i in items if i["category"].lower() == category.lower()]
    if query:
        items = [i for i in items if _similarity(query, i["name"]) >= 0.4]
    return items


@lru_cache
def _get_collection():
    """Embed each menu item (name + description) into a persistent Chroma collection."""
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        "menu_items", metadata={"hnsw:space": "cosine"}
    )

    items = load_menu()
    if collection.count() == len(items):
        return collection

    texts = [f"{item['name']}. {item['description'] or ''}".strip() for item in items]
    collection.upsert(
        ids=[item["id"] for item in items],
        embeddings=embed_texts(texts),
        documents=texts,
        metadatas=[{"name": item["name"]} for item in items],
    )
    return collection


def search_menu_semantic(query: str, k: int | None = None) -> list[dict]:
    """RAG-style retrieval: embed query with the IndoBERT encoder, find nearest menu items."""
    settings = get_settings()
    k = k or settings.menu_rag_top_k
    collection = _get_collection()

    result = collection.query(query_embeddings=[embed_text(query)], n_results=k)
    ids = result["ids"][0] if result["ids"] else []

    by_id = {item["id"]: item for item in load_menu()}
    return [by_id[item_id] for item_id in ids if item_id in by_id]
