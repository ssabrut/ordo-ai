"""Shared fixtures for ordo-ai tests."""
import pytest

MENU_ITEMS = [
    {"id": "d00000000000000000000003", "name": "Ayam Bakar", "category": "Makanan", "price": 30000, "description": "", "available": True, "make_duration": 15},
    {"id": "d0000000000000000000000a", "name": "Ayam Goreng", "category": "Makanan", "price": 28000, "description": "", "available": True, "make_duration": None},
    {"id": "d00000000000000000000002", "name": "Mie Ayam Bakso", "category": "Makanan", "price": 20000, "description": "", "available": True, "make_duration": 8},
    {"id": "d00000000000000000000009", "name": "Mie Goreng Spesial", "category": "Makanan", "price": 22000, "description": "", "available": True, "make_duration": None},
    {"id": "d00000000000000000000007", "name": "Nasi Goreng Ikan Asin", "category": "Makanan", "price": 27000, "description": "", "available": True, "make_duration": None},
    {"id": "d00000000000000000000008", "name": "Nasi Goreng Seafood", "category": "Makanan", "price": 32000, "description": "", "available": True, "make_duration": None},
    {"id": "d00000000000000000000001", "name": "Nasi Goreng Spesial", "category": "Makanan", "price": 25000, "description": "", "available": True, "make_duration": 10},
    {"id": "d0000000000000000000000b", "name": "Es Jeruk", "category": "Minuman", "price": 6000, "description": "", "available": True, "make_duration": None},
    {"id": "d00000000000000000000005", "name": "Es Teh Manis", "category": "Minuman", "price": 5000, "description": "", "available": True, "make_duration": 2},
    {"id": "d00000000000000000000006", "name": "Jus Alpukat", "category": "Minuman", "price": 15000, "description": "", "available": True, "make_duration": 5},
    {"id": "d0000000000000000000000c", "name": "Jus Mangga", "category": "Minuman", "price": 13000, "description": "", "available": True, "make_duration": None},
]


@pytest.fixture
def menu_items():
    return MENU_ITEMS


@pytest.fixture
def mock_menu(monkeypatch):
    """Patch load_menu to return local MENU_ITEMS without disk I/O."""
    import ordo_ai.tools.menu as menu_mod
    monkeypatch.setattr(menu_mod, "load_menu", lambda: MENU_ITEMS)
    menu_mod.load_menu.cache_clear() if hasattr(menu_mod.load_menu, "cache_clear") else None
    return MENU_ITEMS


@pytest.fixture
def ayam_bakar():
    return MENU_ITEMS[0]  # Ayam Bakar, price=30000


@pytest.fixture
def es_teh():
    return MENU_ITEMS[8]  # Es Teh Manis, price=5000


@pytest.fixture
def cart_with_ayam(ayam_bakar):
    return [{"menu_id": ayam_bakar["id"], "name": ayam_bakar["name"], "price": ayam_bakar["price"], "quantity": 1, "notes": []}]


def make_state(**kwargs):
    """Build a minimal OrderState dict for testing."""
    base = {"cart": [], "entities": [], "node_timings": {}}
    base.update(kwargs)
    return base
