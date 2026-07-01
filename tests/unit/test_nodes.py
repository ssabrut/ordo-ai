"""Unit tests for pure/lightweight node logic (no model loading)."""
import pytest

from tests.conftest import make_state


# ---------------------------------------------------------------------------
# normalize node
# ---------------------------------------------------------------------------
class TestNormalizeNode:
    def test_lowercase(self):
        from ordo_ai.nodes.normalize import normalize
        assert normalize("NASI GORENG") == "nasi goreng"

    def test_removes_punctuation(self):
        from ordo_ai.nodes.normalize import normalize
        result = normalize("nasi, goreng!")
        assert "," not in result and "!" not in result

    def test_numbers_to_words(self):
        from ordo_ai.nodes.normalize import normalize
        result = normalize("pesan 2 ayam goreng")
        assert "dua" in result
        assert "2" not in result

    def test_collapses_whitespace(self):
        from ordo_ai.nodes.normalize import normalize
        assert normalize("nasi  goreng   spesial") == "nasi goreng spesial"

    def test_run_returns_normalized_text_key(self):
        from ordo_ai.nodes.normalize import run
        state = make_state(raw_text="Pesan 1 Ayam Bakar!")
        result = run(state)
        assert "normalized_text" in result
        assert "satu" in result["normalized_text"]
        assert "Pesan" not in result["normalized_text"]


# ---------------------------------------------------------------------------
# router node
# ---------------------------------------------------------------------------
class TestRouter:
    def _state(self, intent, confidence):
        return make_state(intent=intent, intent_confidence=confidence)

    def test_high_confidence_returns_confident(self):
        from ordo_ai.nodes.router import route_on_confidence
        assert route_on_confidence(self._state("order_create", 0.95)) == "confident"

    def test_low_confidence_returns_low_confidence(self):
        from ordo_ai.nodes.router import route_on_confidence
        assert route_on_confidence(self._state("order_create", 0.3)) == "low_confidence"

    def test_exactly_at_threshold_is_confident(self):
        from ordo_ai.nodes.router import route_on_confidence
        from ordo_ai.config import get_settings
        threshold = get_settings().intent_confidence_threshold
        assert route_on_confidence(self._state("order_create", threshold)) == "confident"

    @pytest.mark.parametrize("intent,expected_agent", [
        ("order_create", "order_agent"),
        ("order_add_item", "order_agent"),
        ("order_remove_item", "order_agent"),
        ("order_cancel", "order_agent"),
        ("order_modify_quantity", "order_agent"),
        ("order_swap", "order_agent"),
        ("menu_inquiry", "menu_agent"),
        ("confirm", "dialog_agent"),
        ("deny", "dialog_agent"),
        ("repeat_request", "dialog_agent"),
        ("chitchat_oos", "fallback_agent"),
        ("unknown_intent", "fallback_agent"),
    ])
    def test_route_to_agent(self, intent, expected_agent):
        from ordo_ai.nodes.router import route_to_agent
        assert route_to_agent(self._state(intent, 0.9)) == expected_agent

    def test_clarify_sets_needs_clarification(self):
        from ordo_ai.nodes.router import clarify
        state = self._state("order_create", 0.4)
        result = clarify(state)
        assert result["needs_clarification"] is True
        assert "clarification_message" in result
        assert "0.40" in result["clarification_message"]


# ---------------------------------------------------------------------------
# cart tools
# ---------------------------------------------------------------------------
class TestCartTools:
    def test_find_cart_index_exact_match(self, cart_with_ayam):
        from ordo_ai.tools.cart import find_cart_index
        assert find_cart_index(cart_with_ayam, "Ayam Bakar") == 0

    def test_find_cart_index_case_insensitive(self, cart_with_ayam):
        from ordo_ai.tools.cart import find_cart_index
        assert find_cart_index(cart_with_ayam, "ayam bakar") == 0

    def test_find_cart_index_missing(self, cart_with_ayam):
        from ordo_ai.tools.cart import find_cart_index
        assert find_cart_index(cart_with_ayam, "Es Teh Manis") is None

    def test_find_cart_index_fuzzy_match(self, cart_with_ayam):
        from ordo_ai.tools.cart import find_cart_index_fuzzy
        assert find_cart_index_fuzzy(cart_with_ayam, "ayam bakar") == 0

    def test_find_cart_index_fuzzy_nya_suffix(self, cart_with_ayam):
        from ordo_ai.tools.cart import find_cart_index_fuzzy
        assert find_cart_index_fuzzy(cart_with_ayam, "ayam bakarnya") == 0

    def test_find_cart_index_fuzzy_no_false_positive(self, cart_with_ayam):
        from ordo_ai.tools.cart import find_cart_index_fuzzy
        assert find_cart_index_fuzzy(cart_with_ayam, "mie goreng") is None

    def test_add_item_new(self, ayam_bakar):
        from ordo_ai.tools.cart import add_item
        cart, msg = add_item([], ayam_bakar, quantity=2)
        assert len(cart) == 1
        assert cart[0]["quantity"] == 2
        assert cart[0]["name"] == "Ayam Bakar"
        assert "ditambahkan" in msg

    def test_add_item_merges_existing(self, ayam_bakar):
        from ordo_ai.tools.cart import add_item
        cart, _ = add_item([], ayam_bakar, quantity=1)
        cart, msg = add_item(cart, ayam_bakar, quantity=2)
        assert len(cart) == 1
        assert cart[0]["quantity"] == 3
        assert "total 3" in msg

    def test_add_item_with_notes(self, ayam_bakar):
        from ordo_ai.tools.cart import add_item
        cart, _ = add_item([], ayam_bakar, quantity=1, notes=["tanpa sambal"])
        assert cart[0]["notes"] == ["tanpa sambal"]

    def test_add_item_does_not_mutate_input(self, ayam_bakar):
        from ordo_ai.tools.cart import add_item
        original = []
        add_item(original, ayam_bakar, quantity=1)
        assert original == []


# ---------------------------------------------------------------------------
# menu tools (no disk I/O — patch load_menu)
# ---------------------------------------------------------------------------
class TestMenuTools:
    def test_find_menu_item_exact(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_item
        result = find_menu_item("Ayam Bakar")
        assert result is not None
        assert result["name"] == "Ayam Bakar"

    def test_find_menu_item_fuzzy(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_item
        result = find_menu_item("ayam bakar")
        assert result is not None
        assert result["name"] == "Ayam Bakar"

    def test_find_menu_item_below_threshold(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_item
        result = find_menu_item("pizza hawaii")
        assert result is None

    def test_find_menu_items_single_unambiguous(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_items
        results = find_menu_items("Ayam Bakar")
        assert len(results) == 1
        assert results[0]["name"] == "Ayam Bakar"

    def test_find_menu_items_ambiguous(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_items
        results = find_menu_items("nasi goreng")
        assert len(results) > 1
        names = [r["name"] for r in results]
        assert any("Nasi Goreng" in n for n in names)

    def test_find_menu_items_not_found(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_items
        results = find_menu_items("burger keju")
        assert results == []

    def test_find_menu_items_no_cross_word_false_positive(self, mock_menu):
        from ordo_ai.tools.menu import find_menu_items
        results = find_menu_items("mie goreng")
        names = [r["name"] for r in results]
        assert "Ayam Goreng" not in names
