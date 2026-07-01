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
    """Fuzzy-match an entity span against the menu catalog. Returns single best match."""
    items = load_menu()
    best, best_score = None, 0.0
    for item in items:
        score = _similarity(name, item["name"])
        if score > best_score:
            best, best_score = item, score
    if best is not None and best_score >= threshold:
        return best
    return None


def find_menu_items(name: str, threshold: float = 0.55, band: float = 0.02) -> list[dict]:
    """Return candidates within `band` of the top similarity score.

    Also filters out items that don't share any query words (avoids cross-word
    false positives like 'nasi goreng' matching 'Ayam Goreng').

    Returns [] if nothing clears `threshold`.
    Returns [single] when one item is clearly best.
    Returns multiple when top candidates tie (ambiguous query like 'nasi goreng').
    """
    items = load_menu()
    query_words = set(name.lower().split())
    scored = sorted(
        [(item, _similarity(name, item["name"])) for item in items],
        key=lambda x: x[1],
        reverse=True,
    )
    if not scored or scored[0][1] < threshold:
        return []
    # filter to items containing all query words, scored above threshold
    subset_scored = [
        (item, score) for item, score in scored
        if score >= threshold and query_words.issubset(set(item["name"].lower().split()))
    ]
    if subset_scored:
        top_score = subset_scored[0][1]
        return [item for item, score in subset_scored if score >= top_score - band]
    # no item contains all query words — fall back to single raw best match
    best_item, best_score = scored[0]
    return [best_item] if best_score >= threshold else []


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
