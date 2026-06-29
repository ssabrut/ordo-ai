import json
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

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
