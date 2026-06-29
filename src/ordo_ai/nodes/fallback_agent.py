from ordo_ai.state.schemas import OrderState


def run(state: OrderState) -> OrderState:
    return {
        "agent_response": (
            "Maaf, saya hanya bisa membantu pemesanan makanan dan minuman. "
            "Ada yang ingin Anda pesan?"
        )
    }
