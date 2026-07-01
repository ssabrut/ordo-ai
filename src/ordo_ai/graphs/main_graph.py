import time
import logging
from langgraph.graph import END, StateGraph

from ordo_ai.nodes import (dialog_agent, disfluency, fallback_agent, intent,
                           menu_agent, ner, normalize, order_agent, router,
                           stt)
from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)


def _timed(name: str, fn):
    def wrapper(state: OrderState) -> OrderState:
        t0 = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - t0
        logger.debug("timing: node=%r elapsed=%.4fs", name, elapsed)
        existing = state.get("node_timings") or {}
        return {**(result or {}), "node_timings": {**existing, name: elapsed}}
    return wrapper


def build_graph():
    builder = StateGraph(OrderState)

    builder.add_node("stt", _timed("stt", stt.run))
    builder.add_node("normalize", _timed("normalize", normalize.run))
    builder.add_node("disfluency", _timed("disfluency", disfluency.run))
    builder.add_node("ner", _timed("ner", ner.run))
    builder.add_node("intent", _timed("intent", intent.run))
    builder.add_node("clarify", _timed("clarify", router.clarify))
    builder.add_node("order_agent", _timed("order_agent", order_agent.run))
    builder.add_node("menu_agent", _timed("menu_agent", menu_agent.run))
    builder.add_node("dialog_agent", _timed("dialog_agent", dialog_agent.run))
    builder.add_node("fallback_agent", _timed("fallback_agent", fallback_agent.run))

    builder.set_entry_point("stt")
    builder.add_edge("stt", "normalize")
    builder.add_edge("normalize", "disfluency")
    builder.add_edge("disfluency", "ner")
    builder.add_edge("ner", "intent")

    def route_after_intent(state: OrderState) -> str:
        if router.route_on_confidence(state) == "low_confidence":
            return "clarify"
        return router.route_to_agent(state)

    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "clarify": "clarify",
            "order_agent": "order_agent",
            "menu_agent": "menu_agent",
            "dialog_agent": "dialog_agent",
            "fallback_agent": "fallback_agent",
        },
    )
    builder.add_edge("order_agent", END)
    builder.add_edge("menu_agent", END)
    builder.add_edge("dialog_agent", END)
    builder.add_edge("fallback_agent", END)
    builder.add_edge("clarify", END)

    return builder


def compile_graph(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer)


graph = compile_graph()
