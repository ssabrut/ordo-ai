from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"

    disfluency_model_path: str = "models/indobert-disfluency-bio-final"
    ner_model_path: str = "models/indobert-ner-bio-final"
    intent_model_path: str = "models/indobert-intent-final"

    stt_model_size: str = "large-v3-turbo"
    stt_language: str = "id"
    stt_device: str = "cpu"
    stt_compute_type: str = "float32"
    stt_sample_rate: int = 16_000
    stt_input_device_index: int | None = None
    stt_input_device_name: str | None = "MacBook Pro Microphone"

    wake_word: str = "hey_jarvis"
    wake_word_model_path: str | None = None  # None -> openwakeword's hey_jarvis_v0.1.onnx
    wake_word_sensitivity: float = 0.6
    wake_word_timeout_s: float = 5.0

    max_seq_length: int = 64
    intent_confidence_threshold: float = 0.6

    chroma_persist_dir: str = "data/chroma_menu"
    menu_rag_top_k: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
