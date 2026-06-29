import logging

from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)


def run(state: OrderState) -> OrderState:
    result = {
        "agent_response": (
            "Maaf, saya hanya bisa membantu pemesanan makanan dan minuman. "
            "Ada yang ingin Anda pesan?"
        )
    }
    logger.debug("fallback_agent: intent=%r result=%r", state.get("intent"), result)
    return result
