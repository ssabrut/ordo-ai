from typing import Literal

from ordo_ai.config import get_settings
from ordo_ai.state.schemas import OrderState


def route_on_confidence(state: OrderState) -> Literal["confident", "low_confidence"]:
    settings = get_settings()
    if state["intent_confidence"] >= settings.intent_confidence_threshold:
        return "confident"
    return "low_confidence"


def clarify(state: OrderState) -> OrderState:
    return {
        "needs_clarification": True,
        "clarification_message": (
            f"Maaf, saya tidak yakin maksud Anda (intent={state['intent']!r}, "
            f"confidence={state['intent_confidence']:.2f}). Bisa diulang?"
        ),
    }
