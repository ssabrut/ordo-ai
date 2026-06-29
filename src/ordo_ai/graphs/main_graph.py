from langgraph.graph import END, StateGraph

from ordo_ai.nodes import disfluency, intent, ner, normalize, router, stt
from ordo_ai.state.schemas import OrderState


def build_graph():
    builder = StateGraph(OrderState)

    builder.add_node("stt", stt.run)
    builder.add_node("normalize", normalize.run)
    builder.add_node("disfluency", disfluency.run)
    builder.add_node("ner", ner.run)
    builder.add_node("intent", intent.run)
    builder.add_node("clarify", router.clarify)

    builder.set_entry_point("stt")
    builder.add_edge("stt", "normalize")
    builder.add_edge("normalize", "disfluency")
    builder.add_edge("disfluency", "ner")
    builder.add_edge("ner", "intent")
    builder.add_conditional_edges(
        "intent",
        router.route_on_confidence,
        {"confident": END, "low_confidence": "clarify"},
    )
    builder.add_edge("clarify", END)

    return builder


def compile_graph(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer)


graph = compile_graph()
