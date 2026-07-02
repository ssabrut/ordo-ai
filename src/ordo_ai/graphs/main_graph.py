import time
import logging
from pathlib import Path

from langgraph.graph import END, StateGraph

from ordo_ai.nodes import (dialog_agent, disfluency, fallback_agent, intent,
                           menu_agent, ner, normalize, order_agent, router,
                           stt)
from ordo_ai.state.schemas import OrderState

_AGENT_MODULES = {
    "order_agent": order_agent,
    "menu_agent": menu_agent,
    "dialog_agent": dialog_agent,
    "fallback_agent": fallback_agent,
}

logger = logging.getLogger(__name__)

_EXPORT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "graph.png"


def _timed(name: str, fn):
    def wrapper(state: OrderState) -> OrderState:
        t0 = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - t0
        logger.debug("timing: node=%r elapsed=%.4fs", name, elapsed)
        existing = state.get("node_timings") or {}
        return {**(result or {}), "node_timings": {**existing, name: elapsed}}
    return wrapper


def multi_agent_dispatch(state: OrderState) -> OrderState:
    """Sequentially run the agent mapped to each predicted intent against the
    running state, threading cart/pending_item/last_discussed_item mutations
    from one call into the next and concatenating agent_response. Stops early
    if an agent sets needs_clarification, pausing any remaining intents.
    """
    working_state = dict(state)
    responses = []

    for current_intent in state["intents"]:
        agent_name = router.route_to_agent(current_intent)
        agent_fn = _AGENT_MODULES[agent_name].run
        working_state["intent"] = current_intent

        result = agent_fn(working_state) or {}
        working_state.update(result)

        if result.get("agent_response"):
            responses.append(result["agent_response"])

        if working_state.get("needs_clarification"):
            break

    working_state["agent_response"] = " ".join(responses)
    return working_state


def build_graph():
    builder = StateGraph(OrderState)

    builder.add_node("stt", _timed("stt", stt.run))
    builder.add_node("normalize", _timed("normalize", normalize.run))
    builder.add_node("disfluency", _timed("disfluency", disfluency.run))
    builder.add_node("ner", _timed("ner", ner.run))
    builder.add_node("intent", _timed("intent", intent.run))
    builder.add_node("clarify", _timed("clarify", router.clarify))
    builder.add_node("multi_agent_dispatch", _timed("multi_agent_dispatch", multi_agent_dispatch))

    builder.set_entry_point("stt")
    builder.add_edge("stt", "normalize")
    builder.add_edge("normalize", "disfluency")
    builder.add_edge("disfluency", "ner")
    builder.add_edge("ner", "intent")

    def route_after_intent(state: OrderState) -> str:
        if router.route_on_confidence(state) == "low_confidence":
            return "clarify"
        return "multi_agent_dispatch"

    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "clarify": "clarify",
            "multi_agent_dispatch": "multi_agent_dispatch",
        },
    )
    builder.add_edge("multi_agent_dispatch", END)
    builder.add_edge("clarify", END)

    return builder


def _export_png(compiled) -> None:
    try:
        png_bytes = compiled.get_graph().draw_mermaid_png()
        _EXPORT_PATH.write_bytes(png_bytes)
        logger.info("main_graph: exported graph PNG to %r", str(_EXPORT_PATH))
    except Exception:
        logger.exception("main_graph: failed to export graph PNG to %r", str(_EXPORT_PATH))


def compile_graph(checkpointer=None):
    compiled = build_graph().compile(checkpointer=checkpointer)
    _export_png(compiled)
    return compiled


graph = compile_graph()
