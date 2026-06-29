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
    "menu_inquiry": "menu_agent",
    "confirm": "dialog_agent",
    "deny": "dialog_agent",
    "repeat_request": "dialog_agent",
    "chitchat_oos": "fallback_agent",
}


def route_on_confidence(state: OrderState) -> Literal["confident", "low_confidence"]:
    settings = get_settings()
    if state["intent_confidence"] >= settings.intent_confidence_threshold:
        return "confident"
    return "low_confidence"


def route_to_agent(state: OrderState) -> AgentName:
    agent = _INTENT_TO_AGENT.get(state["intent"], "fallback_agent")
    logger.debug("router: intent=%r -> agent=%r", state["intent"], agent)
    return agent


def clarify(state: OrderState) -> OrderState:
    result = {
        "needs_clarification": True,
        "clarification_message": (
            f"Maaf, saya tidak yakin maksud Anda (intent={state['intent']!r}, "
            f"confidence={state['intent_confidence']:.2f}). Bisa diulang?"
        ),
    }
    logger.debug("clarify: %r", result)
    return result
