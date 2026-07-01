"""Unit tests for order_agent and dialog_agent node logic (mocked menu)."""
import pytest

from tests.conftest import make_state, MENU_ITEMS

AYAM_BAKAR = MENU_ITEMS[0]   # id=d00000000000000000000003, price=30000
AYAM_GORENG = MENU_ITEMS[1]  # id=d0000000000000000000000a, price=28000
MIE_GORENG = MENU_ITEMS[3]   # Mie Goreng Spesial, price=22000
ES_TEH = MENU_ITEMS[8]       # Es Teh Manis, price=5000

# ---------------------------------------------------------------------------
# order_agent
# ---------------------------------------------------------------------------
class TestOrderAgentAdd:
    def test_add_single_dish(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="tambah ayam bakar",
            entities=[{"text": "ayam bakar", "label": "DISH", "start": 7, "end": 17}],
        )
        result = run(state)
        assert len(result["cart"]) == 1
        assert result["cart"][0]["name"] == "Ayam Bakar"
        assert result["cart"][0]["quantity"] == 1

    def test_add_with_quantity(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="pesan dua ayam bakar",
            entities=[
                {"text": "dua", "label": "QUANTITY", "start": 6, "end": 9},
                {"text": "ayam bakar", "label": "DISH", "start": 10, "end": 20},
            ],
        )
        result = run(state)
        assert result["cart"][0]["quantity"] == 2

    def test_add_unknown_item(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="pesan pizza",
            entities=[{"text": "pizza", "label": "DISH", "start": 6, "end": 11}],
        )
        result = run(state)
        assert result["cart"] == []
        assert "tidak tersedia" in result["agent_response"]

    def test_add_ambiguous_triggers_pending(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        # "nasi goreng" matches multiple candidates
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="pesan nasi goreng",
            entities=[{"text": "nasi goreng", "label": "DISH", "start": 6, "end": 17}],
        )
        result = run(state)
        assert result.get("needs_clarification") is True
        assert result.get("pending_item") is not None
        assert len(result["pending_item"]["candidates"]) > 1

    def test_add_merges_same_item(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        existing_cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="tambah lagi ayam bakar",
            entities=[{"text": "ayam bakar", "label": "DISH", "start": 12, "end": 22}],
            cart=existing_cart,
        )
        result = run(state)
        assert len(result["cart"]) == 1
        assert result["cart"][0]["quantity"] == 2


class TestOrderAgentRemove:
    def test_remove_existing_item(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        state = make_state(
            intent="order_remove_item",
            intent_confidence=0.95,
            repaired_text="hapus ayam bakar",
            entities=[{"text": "ayam bakar", "label": "DISH", "start": 6, "end": 16}],
            cart=cart,
        )
        result = run(state)
        assert result["cart"] == []
        assert "dihapus" in result["agent_response"]

    def test_remove_nonexistent_item(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        state = make_state(
            intent="order_remove_item",
            intent_confidence=0.95,
            repaired_text="hapus es teh",
            entities=[{"text": "es teh", "label": "DISH", "start": 6, "end": 12}],
            cart=[],
        )
        result = run(state)
        assert "tidak ditemukan" in result["agent_response"]


class TestOrderAgentCancel:
    def test_cancel_clears_cart(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 2, "notes": []}]
        state = make_state(
            intent="order_cancel",
            intent_confidence=0.95,
            repaired_text="batalkan pesanan",
            entities=[],
            cart=cart,
        )
        result = run(state)
        assert result["cart"] == []
        assert "dibatalkan" in result["agent_response"]


class TestOrderAgentModifyQuantity:
    def test_modify_quantity(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        state = make_state(
            intent="order_modify_quantity",
            intent_confidence=0.95,
            repaired_text="ubah ayam bakar jadi tiga",
            entities=[
                {"text": "tiga", "label": "QUANTITY", "start": 21, "end": 25},
                {"text": "ayam bakar", "label": "DISH", "start": 5, "end": 15},
            ],
            cart=cart,
        )
        result = run(state)
        assert result["cart"][0]["quantity"] == 3


class TestOrderAgentSwap:
    def test_swap_unambiguous(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        state = make_state(
            intent="order_swap",
            intent_confidence=0.95,
            repaired_text="ganti ayam bakar dengan es teh manis",
            entities=[
                {"text": "ayam bakar", "label": "DISH", "start": 6, "end": 16},
                {"text": "es teh manis", "label": "DRINK", "start": 25, "end": 37},
            ],
            cart=cart,
        )
        result = run(state)
        names = [i["name"] for i in result["cart"]]
        assert "Ayam Bakar" not in names
        assert "Es Teh Manis" in names

    def test_swap_missing_second_item(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        state = make_state(
            intent="order_swap",
            intent_confidence=0.95,
            repaired_text="ganti ayam bakar",
            entities=[{"text": "ayam bakar", "label": "DISH", "start": 6, "end": 16}],
            cart=cart,
        )
        result = run(state)
        assert "sebutkan" in result["agent_response"]


class TestOrderAgentLastDiscussed:
    def test_consumes_last_discussed_on_itu(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="itu satu",
            entities=[{"text": "satu", "label": "QUANTITY", "start": 4, "end": 8}],
            last_discussed_item=AYAM_BAKAR,
        )
        result = run(state)
        assert any(i["name"] == "Ayam Bakar" for i in result["cart"])
        assert result.get("last_discussed_item") is None

    def test_no_consume_without_reference_word(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        state = make_state(
            intent="order_add_item",
            intent_confidence=0.95,
            repaired_text="tambah mie ayam bakso",
            entities=[{"text": "mie ayam bakso", "label": "DISH", "start": 7, "end": 21}],
            last_discussed_item=AYAM_BAKAR,
        )
        result = run(state)
        names = [i["name"] for i in result["cart"]]
        assert "Ayam Bakar" not in names


class TestOrderAgentResolvePending:
    def test_resolves_pending_via_entity(self, mock_menu):
        from ordo_ai.nodes.order_agent import run
        from tests.conftest import MENU_ITEMS
        candidates = [m for m in MENU_ITEMS if "Nasi Goreng" in m["name"]]
        pending = {"name": "nasi goreng", "quantity": 1, "notes": [], "candidates": candidates}
        state = make_state(
            intent="confirm",
            intent_confidence=0.9,
            repaired_text="yang spesial",
            entities=[{"text": "spesial", "label": "DISH", "start": 5, "end": 12}],
            needs_clarification=True,
            pending_item=pending,
        )
        result = run(state)
        # spesial keyword should pick one of the "Spesial" variants
        assert result.get("needs_clarification") is False
        assert result.get("pending_item") is None
        assert len(result["cart"]) == 1


# ---------------------------------------------------------------------------
# dialog_agent
# ---------------------------------------------------------------------------
class TestDialogAgentDeny:
    def test_deny_clears_cart(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        state = make_state(intent="deny", intent_confidence=0.9, repaired_text="tidak", cart=cart)
        result = run(state)
        assert result["cart"] == []
        assert result["needs_clarification"] is False


class TestDialogAgentRepeat:
    def test_repeat_with_empty_cart(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        state = make_state(intent="repeat_request", intent_confidence=0.9, repaired_text="ulangi", cart=[])
        result = run(state)
        assert "kosong" in result["agent_response"]

    def test_repeat_with_items(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 2, "notes": []}]
        state = make_state(intent="repeat_request", intent_confidence=0.9, repaired_text="ulangi", cart=cart)
        result = run(state)
        assert "Ayam Bakar" in result["agent_response"]
        assert "2x" in result["agent_response"]


class TestDialogAgentConfirmPending:
    def test_confirm_with_ordinal(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        from tests.conftest import MENU_ITEMS
        candidates = [m for m in MENU_ITEMS if "Nasi Goreng" in m["name"]]
        pending = {"name": "nasi goreng", "quantity": 1, "notes": [], "candidates": candidates}
        state = make_state(
            intent="confirm",
            intent_confidence=0.9,
            repaired_text="yang pertama",
            needs_clarification=True,
            pending_item=pending,
        )
        result = run(state)
        assert result["needs_clarification"] is False
        assert result["pending_item"] is None
        assert result["cart"][0]["name"] == candidates[0]["name"]

    def test_confirm_with_name_match(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        from tests.conftest import MENU_ITEMS
        candidates = [m for m in MENU_ITEMS if "Nasi Goreng" in m["name"]]
        pending = {"name": "nasi goreng", "quantity": 1, "notes": [], "candidates": candidates}
        state = make_state(
            intent="confirm",
            intent_confidence=0.9,
            repaired_text="nasi goreng seafood",
            needs_clarification=True,
            pending_item=pending,
        )
        result = run(state)
        assert result["needs_clarification"] is False
        assert "Seafood" in result["cart"][0]["name"]

    def test_confirm_no_pick_re_asks(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        from tests.conftest import MENU_ITEMS
        candidates = [m for m in MENU_ITEMS if "Nasi Goreng" in m["name"]]
        pending = {"name": "nasi goreng", "quantity": 1, "notes": [], "candidates": candidates}
        state = make_state(
            intent="confirm",
            intent_confidence=0.9,
            repaired_text="iya",
            needs_clarification=True,
            pending_item=pending,
        )
        result = run(state)
        assert result["needs_clarification"] is True
        assert "Pilih nomor" in result["agent_response"]


class TestDialogAgentConfirmLastDiscussed:
    def test_confirm_adds_last_discussed(self, mock_menu):
        from ordo_ai.nodes.dialog_agent import run
        state = make_state(
            intent="confirm",
            intent_confidence=0.9,
            repaired_text="iya mau",
            entities=[],
            last_discussed_item=AYAM_BAKAR,
        )
        result = run(state)
        assert any(i["name"] == "Ayam Bakar" for i in result["cart"])
        assert result.get("last_discussed_item") is None
