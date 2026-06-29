import logging
import os
from functools import lru_cache
from typing import Callable

import openwakeword
from RealtimeSTT import AudioToTextRecorder

from ordo_ai.config import get_settings
from ordo_ai.state.schemas import OrderState

logger = logging.getLogger(__name__)


def _default_wake_word_model_path() -> str:
    return os.path.join(
        os.path.dirname(openwakeword.__file__),
        "resources",
        "models",
        "hey_jarvis_v0.1.onnx",
    )


@lru_cache
def find_input_device_index(name_substring: str) -> int:
    """Look up a PyAudio input device index by case-insensitive substring."""
    import pyaudio

    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if (
                info["maxInputChannels"] > 0
                and name_substring.lower() in info["name"].lower()
            ):
                return i
    finally:
        p.terminate()
    raise ValueError(f"no input device matching {name_substring!r}")


def make_wakeword_recorder(
    on_wakeword_detected: Callable[[], None] | None = None,
    on_realtime_update: Callable[[str], None] | None = None,
    on_stabilized: Callable[[str], None] | None = None,
    on_recorded_chunk: Callable[[bytes], None] | None = None,
) -> AudioToTextRecorder:
    """Mic stays idle (no transcription) until the wake word fires, then
    transcribes one utterance and re-arms. See notebooks/1_speech_to_text.ipynb.
    """
    settings = get_settings()
    openwakeword.utils.download_models()  # no-op if already cached

    wake_word_model_path = (
        settings.wake_word_model_path or _default_wake_word_model_path()
    )

    input_device_index = settings.stt_input_device_index
    if input_device_index is None and settings.stt_input_device_name:
        input_device_index = find_input_device_index(settings.stt_input_device_name)

    return AudioToTextRecorder(
        model=settings.stt_model_size,
        language=settings.stt_language,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        sample_rate=settings.stt_sample_rate,
        input_device_index=input_device_index,
        enable_realtime_transcription=on_realtime_update is not None,
        realtime_model_type=settings.stt_model_size,
        realtime_processing_pause=0.2,
        on_realtime_transcription_update=on_realtime_update,
        on_realtime_transcription_stabilized=on_stabilized,
        wakeword_backend="oww",
        openwakeword_model_paths=wake_word_model_path,
        wake_words_sensitivity=settings.wake_word_sensitivity,
        wake_word_activation_delay=0.0,
        wake_word_timeout=settings.wake_word_timeout_s,
        on_wakeword_detected=on_wakeword_detected,
        on_recorded_chunk=on_recorded_chunk,
        spinner=False,
    )


def run(state: OrderState) -> OrderState:
    """Passthrough — `raw_text` is captured upstream by the wake-word loop
    (see scripts/run_voice_session.py) and passed in as the graph input.
    Graph invocation happens per-utterance, so there's no listening to do here.
    """
    logger.debug("stt: raw_text=%r", state["raw_text"])
    return {"raw_text": state["raw_text"]}
