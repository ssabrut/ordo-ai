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
    result = {}
    for update in graph.stream({"raw_text": text}, stream_mode="updates"):
        for node_name, node_output in update.items():
            print(f"  [{node_name}] {node_output}")
            result.update(node_output)

    if result.get("needs_clarification"):
        print(f"  -> {result['clarification_message']}")
    else:
        print(f"  -> intent={result['intent']} (conf={result['intent_confidence']:.2f})")
        print(f"  -> entities={result['entities']}")
        print(f"  -> agent_response={result.get('agent_response')}")

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
