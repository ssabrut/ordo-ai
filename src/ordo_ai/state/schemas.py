from typing import TypedDict


class EntitySpan(TypedDict):
    text: str
    label: str
    start: int
    end: int


class CartItem(TypedDict):
    menu_id: str
    name: str
    price: int
    quantity: int
    notes: list[str]


class OrderState(TypedDict, total=False):
    audio: bytes
    raw_text: str
    normalized_text: str
    disfluency_tags: list[str]
    repaired_text: str
    entities: list[EntitySpan]
    intent: str
    intent_confidence: float
    intent_probs: dict[str, float]
    needs_clarification: bool
    clarification_message: str
    cart: list[CartItem]
    agent_response: str
