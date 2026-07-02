import logging
from typing import Literal

from ordo_ai.config import get_settings
from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)

AgentName = Literal["order_agent", "menu_agent", "dialog_agent", "fallback_agent"]

_INTENT_TO_AGENT: dict[str, AgentName] = {
    "order_create": "order_agent",
    "order_add_item": "order_agent",
    "order_remove_item": "order_agent",
    "order_cancel": "order_agent",
    "order_modify_quantity": "order_agent",
    "order_swap": "order_agent",
    "menu_inquiry": "menu_agent",
    "confirm": "dialog_agent",
    "deny": "dialog_agent",
    "repeat_request": "dialog_agent",
    "chitchat_oos": "fallback_agent",
}


def route_on_confidence(state: OrderState) -> Literal["confident", "low_confidence"]:
    settings = get_settings()
    confidences = state["intent_confidences"]
    if any(c < settings.intent_confidence_threshold for c in confidences.values()):
        return "low_confidence"
    return "confident"


def route_to_agent(intent: str) -> AgentName:
    agent = _INTENT_TO_AGENT.get(intent, "fallback_agent")
    logger.debug("router: intent=%r -> agent=%r", intent, agent)
    return agent


def clarify(state: OrderState) -> OrderState:
    summary = ", ".join(
        f"{intent}={conf:.2f}" for intent, conf in state["intent_confidences"].items()
    )
    result = {
        "needs_clarification": True,
        "clarification_message": (
            f"Maaf, saya tidak yakin maksud Anda ({summary}). Bisa diulang?"
        ),
    }
    logger.debug("clarify: %r", result)
    return result
