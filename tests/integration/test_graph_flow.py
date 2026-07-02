"""Integration tests for the full LangGraph flow.

Heavy model nodes (STT, disfluency, NER, intent) are replaced with lightweight
stubs so these tests run without GPU/model weights. The stubs write the same
state keys the real nodes would write, letting downstream nodes and the router
behave exactly as in production.
"""
import pytest

from tests.conftest import make_state, MENU_ITEMS

AYAM_BAKAR = MENU_ITEMS[0]
ES_TEH = MENU_ITEMS[8]
NASI_GORENG_SPESIAL = MENU_ITEMS[6]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stub_stt(state):
    return {}


def _stub_disfluency(state):
    text = state.get("normalized_text", "")
    return {"disfluency_tags": ["O"] * len(text.split()), "repaired_text": text}


def _make_intent_stub(intent, confidence: float = 0.95, probs: dict | None = None):
    """`intent` may be a single label string or a list of labels (multi-intent)."""
    intents = [intent] if isinstance(intent, str) else list(intent)

    def _stub(state):
        return {
            "intents": intents,
            "intent_confidences": {i: confidence for i in intents},
            "intent_probs": probs or {i: confidence for i in intents},
        }
    return _stub


def _make_ner_stub(entities: list):
    def _stub(state):
        return {"entities": entities}
    return _stub


def _patch_graph_nodes(monkeypatch, ner_entities: list, intent: str, confidence: float = 0.95):
    """Patch all model-heavy nodes on the compiled graph and return it."""
    import ordo_ai.nodes.stt as stt_mod
    import ordo_ai.nodes.disfluency as disfl_mod
    import ordo_ai.nodes.ner as ner_mod
    import ordo_ai.nodes.intent as intent_mod
    import ordo_ai.tools.menu as menu_mod

    monkeypatch.setattr(stt_mod, "run", _stub_stt)
    monkeypatch.setattr(disfl_mod, "run", _stub_disfluency)
    monkeypatch.setattr(ner_mod, "run", _make_ner_stub(ner_entities))
    monkeypatch.setattr(intent_mod, "run", _make_intent_stub(intent, confidence))
    monkeypatch.setattr(menu_mod, "load_menu", lambda: MENU_ITEMS)

    from ordo_ai.graphs.main_graph import compile_graph
    return compile_graph()


def _invoke(graph, raw_text: str, extra_state: dict | None = None):
    state = make_state(raw_text=raw_text)
    if extra_state:
        state.update(extra_state)
    return graph.invoke(state)


# ---------------------------------------------------------------------------
# Graph routing — low confidence → clarify
# ---------------------------------------------------------------------------
class TestGraphRoutingLowConfidence:
    def test_low_confidence_routes_to_clarify(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="order_create",
            confidence=0.2,
        )
        result = _invoke(graph, "mau pesan sesuatu")
        assert result.get("needs_clarification") is True
        assert "clarification_message" in result

    def test_at_threshold_does_not_clarify(self, monkeypatch):
        from ordo_ai.config import get_settings
        threshold = get_settings().intent_confidence_threshold
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[{"text": "ayam bakar", "label": "DISH", "start": 0, "end": 10}],
            intent="order_add_item",
            confidence=threshold,
        )
        result = _invoke(graph, "pesan ayam bakar")
        assert not result.get("needs_clarification")


# ---------------------------------------------------------------------------
# order_add_item flow
# ---------------------------------------------------------------------------
class TestGraphOrderAdd:
    def test_add_single_item_end_to_end(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[{"text": "ayam bakar", "label": "DISH", "start": 6, "end": 16}],
            intent="order_add_item",
        )
        result = _invoke(graph, "pesan ayam bakar")
        assert len(result["cart"]) == 1
        assert result["cart"][0]["name"] == "Ayam Bakar"
        assert "node_timings" in result
        assert "multi_agent_dispatch" in result["node_timings"]

    def test_add_with_quantity(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[
                {"text": "tiga", "label": "QUANTITY", "start": 0, "end": 4},
                {"text": "ayam bakar", "label": "DISH", "start": 5, "end": 15},
            ],
            intent="order_add_item",
        )
        result = _invoke(graph, "tiga ayam bakar")
        assert result["cart"][0]["quantity"] == 3

    def test_add_two_different_items(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[
                {"text": "ayam bakar", "label": "DISH", "start": 0, "end": 10},
                {"text": "es teh manis", "label": "DRINK", "start": 14, "end": 26},
            ],
            intent="order_add_item",
        )
        result = _invoke(graph, "ayam bakar dan es teh manis")
        names = {i["name"] for i in result["cart"]}
        assert "Ayam Bakar" in names
        assert "Es Teh Manis" in names

    def test_add_unknown_item_gives_response(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[{"text": "soto betawi", "label": "DISH", "start": 0, "end": 11}],
            intent="order_add_item",
        )
        result = _invoke(graph, "soto betawi")
        assert result["cart"] == []
        assert "tidak tersedia" in result.get("agent_response", "")

    def test_ambiguous_item_sets_pending(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[{"text": "nasi goreng", "label": "DISH", "start": 0, "end": 11}],
            intent="order_add_item",
        )
        result = _invoke(graph, "nasi goreng")
        assert result.get("needs_clarification") is True
        assert result["pending_item"] is not None
        assert len(result["pending_item"]["candidates"]) > 1


# ---------------------------------------------------------------------------
# order_cancel flow
# ---------------------------------------------------------------------------
class TestGraphOrderCancel:
    def test_cancel_clears_cart(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="order_cancel",
        )
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        result = _invoke(graph, "batalkan pesanan", extra_state={"cart": cart})
        assert result["cart"] == []


# ---------------------------------------------------------------------------
# order_remove_item flow
# ---------------------------------------------------------------------------
class TestGraphOrderRemove:
    def test_remove_item(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[{"text": "ayam bakar", "label": "DISH", "start": 6, "end": 16}],
            intent="order_remove_item",
        )
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 2, "notes": []}]
        result = _invoke(graph, "hapus ayam bakar", extra_state={"cart": cart})
        assert result["cart"] == []


# ---------------------------------------------------------------------------
# menu_inquiry flow
# ---------------------------------------------------------------------------
class TestGraphMenuInquiry:
    def test_menu_inquiry_routes_to_menu_agent(self, monkeypatch):
        import ordo_ai.nodes.menu_agent as menu_agent_mod

        responses = []

        def _stub_menu_agent(state):
            responses.append("called")
            return {"agent_response": "Menu kami adalah...", "last_discussed_item": AYAM_BAKAR}

        monkeypatch.setattr(menu_agent_mod, "run", _stub_menu_agent)
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="menu_inquiry",
        )
        result = _invoke(graph, "ada menu apa saja")
        assert len(responses) == 1
        assert result["agent_response"] == "Menu kami adalah..."


# ---------------------------------------------------------------------------
# chitchat → fallback flow
# ---------------------------------------------------------------------------
class TestGraphFallback:
    def test_chitchat_routes_to_fallback_agent(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="chitchat_oos",
        )
        result = _invoke(graph, "apa kabar hari ini")
        assert result.get("agent_response") is not None


# ---------------------------------------------------------------------------
# deny flow
# ---------------------------------------------------------------------------
class TestGraphDeny:
    def test_deny_clears_pending_and_cart(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="deny",
        )
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        pending = {"name": "nasi goreng", "quantity": 1, "notes": [], "candidates": MENU_ITEMS[:3]}
        result = _invoke(
            graph, "tidak",
            extra_state={"cart": cart, "pending_item": pending, "needs_clarification": True},
        )
        assert result["cart"] == []
        assert result["needs_clarification"] is False


# ---------------------------------------------------------------------------
# repeat_request flow
# ---------------------------------------------------------------------------
class TestGraphRepeat:
    def test_repeat_includes_cart_items(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="repeat_request",
        )
        cart = [{"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []}]
        result = _invoke(graph, "ulangi", extra_state={"cart": cart})
        assert "Ayam Bakar" in result["agent_response"]


# ---------------------------------------------------------------------------
# confirm disambiguation flow
# ---------------------------------------------------------------------------
class TestGraphConfirmDisambiguation:
    def test_confirm_resolves_pending(self, monkeypatch):
        nasi_goreng_candidates = [m for m in MENU_ITEMS if "Nasi Goreng" in m["name"]]
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[],
            intent="confirm",
        )
        pending = {
            "name": "nasi goreng",
            "quantity": 1,
            "notes": [],
            "candidates": nasi_goreng_candidates,
        }
        # NER stub returns no entities; repaired_text (from disfluency stub) will be
        # the normalized_text, which we set to "yang pertama"
        result = _invoke(
            graph, "yang pertama",
            extra_state={"needs_clarification": True, "pending_item": pending},
        )
        assert result.get("needs_clarification") is False
        assert result["pending_item"] is None
        assert len(result["cart"]) == 1
        assert result["cart"][0]["name"] == nasi_goreng_candidates[0]["name"]


# ---------------------------------------------------------------------------
# multi-intent single-utterance flow
#
# multi_agent_dispatch (main_graph.py) loops over state["intents"] in
# prediction order, calling each mapped agent against the running state so
# add/modify/remove actions from separate intents in one utterance all land.
# Each agent's _group_entities still applies its single intent uniformly to
# whatever entities are present when it runs (order_agent.py `for parsed in
# parsed_items:` loop) — so this test gives each intent its own NER entities
# by invoking the graph would require per-intent entity slicing, which NER
# doesn't do. Instead each intent below acts on the entities relevant to it
# via separate cart starting states / separate entity sets is out of scope;
# this test exercises two intents (add, cancel) which each operate on the
# full (small) entity set unambiguously.
# ---------------------------------------------------------------------------
class TestGraphMultiIntentUtterance:
    def test_add_and_cancel_in_one_utterance(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[
                {"text": "es teh manis", "label": "DRINK", "start": 9, "end": 21},
            ],
            intent=["order_add_item", "order_cancel"],
        )
        cart = [
            {"menu_id": AYAM_BAKAR["id"], "name": "Ayam Bakar", "price": 30000, "quantity": 1, "notes": []},
        ]
        result = _invoke(
            graph,
            "tambahin es teh manis, terus batalin semua pesanan",
            extra_state={"cart": cart},
        )

        # order_add_item runs first (adds Es Teh Manis to the running cart),
        # then order_cancel runs second and clears the whole cart — final
        # state reflects cancel winning since it ran last, per prediction order.
        assert result["cart"] == []
        assert "dibatalkan" in result["agent_response"]

    def test_two_intents_merge_agent_responses(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[
                {"text": "ayam bakar", "label": "DISH", "start": 0, "end": 10},
            ],
            intent=["order_add_item", "repeat_request"],
        )
        result = _invoke(graph, "ayam bakar, terus ulangi pesanan")
        assert len(result["cart"]) == 1
        assert result["cart"][0]["name"] == "Ayam Bakar"
        # dialog_agent's repeat_request response should also be present,
        # concatenated after order_agent's add-item response
        assert "Pesanan Anda" in result["agent_response"]


# ---------------------------------------------------------------------------
# node_timings recorded for every node
# ---------------------------------------------------------------------------
class TestNodeTimings:
    def test_timings_recorded_for_pipeline_nodes(self, monkeypatch):
        graph = _patch_graph_nodes(
            monkeypatch,
            ner_entities=[{"text": "ayam bakar", "label": "DISH", "start": 0, "end": 10}],
            intent="order_add_item",
        )
        result = _invoke(graph, "ayam bakar")
        timings = result.get("node_timings", {})
        for node in ("stt", "normalize", "disfluency", "ner", "intent", "multi_agent_dispatch"):
            assert node in timings, f"timing missing for {node!r}"
            assert timings[node] >= 0
