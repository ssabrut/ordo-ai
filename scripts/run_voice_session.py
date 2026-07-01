"""Continuous wake-word-gated voice session: listens for the wake word, then
runs each transcribed utterance through the full LangGraph order pipeline.

Usage: poetry run python scripts/run_voice_session.py
"""

import threading
import time

from ordo_ai.config import get_settings
from ordo_ai.graphs.main_graph import compile_graph
from ordo_ai.nodes import disfluency, intent, ner
from ordo_ai.nodes.stt import make_wakeword_recorder

settings = get_settings()
graph = compile_graph()
stop_event = threading.Event()

_SESSION_CARRY_KEYS = ("cart", "pending_item", "needs_clarification", "last_discussed_item")
session_state: dict = {}


def warm_up_models():
    """Force model weights off disk now, not on the first wake-word utterance."""
    print("loading models...")
    disfluency._load()
    ner._load()
    intent._load()
    print("models loaded")


def on_wake():
    print(f"[wake] {settings.wake_word} detected, listening...")


def on_final(text: str):
    text = text.strip()
    if not text:
        print(f"listening for wake word ({settings.wake_word})...")
        return

    print(f"[utterance] {text}")
    graph_input = {**{k: v for k, v in session_state.items() if v is not None}, "raw_text": text}
    result = {}
    for update in graph.stream(graph_input, stream_mode="updates"):
        for node_name, node_output in update.items():
            display = {k: v for k, v in node_output.items() if k != "node_timings"}
            print(f"  [{node_name}] {display}")
            result.update(node_output)

    for key in _SESSION_CARRY_KEYS:
        if key in result:
            session_state[key] = result[key]

    if result.get("needs_clarification"):
        print(f"  -> {result.get('clarification_message', result.get('agent_response'))}")
    else:
        print(f"  -> intent={result['intent']} (conf={result['intent_confidence']:.2f})")
        print(f"  -> entities={result['entities']}")
        print(f"  -> agent_response={result.get('agent_response')}")
        print(f"  -> cart={session_state.get('cart', [])}")

    timings = result.get("node_timings", {})
    if timings:
        timing_str = "  -> timings: " + "  ".join(f"{k}={v:.4f}s" for k, v in timings.items())
        print(timing_str)

    print(f"listening for wake word ({settings.wake_word})...")


def main():
    warm_up_models()
    recorder = make_wakeword_recorder(on_wakeword_detected=on_wake)

    def transcription_loop():
        while not stop_event.is_set():
            recorder.text(on_final)

    print(f"listening for wake word ({settings.wake_word})...")
    t = threading.Thread(target=transcription_loop, daemon=True)
    t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        recorder.shutdown()
        t.join(timeout=5)


if __name__ == "__main__":
    main()
