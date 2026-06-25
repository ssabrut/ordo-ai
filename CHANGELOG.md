# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Batch FLEURS (`google/fleurs`, `id_id` validation split) transcription cell: loads dataset via `load_dataset(..., split="validation")` with `Audio(decode=False)`, runs each sample through a separate `WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)` instance, and writes `id`/`audio_path`/`reference`/`hypothesis` records to `data/fleurs_validation_transcripts.jsonl` for offline accuracy/normalization scoring against ground truth.
- Cell to persist `wakeword_debug_log` entries as newline-delimited JSON to `data/transcripts.jsonl` (append mode), capturing live wake-word transcription runs for later evaluation.
- `find_input_device_index(name_substring)`: resolves a PyAudio input device index by case-insensitive substring match against device name (e.g. `"MacBook Pro Microphone"`); raises `ValueError` if no match. Bound to `MACBOOK_MIC_INDEX` and passed as `input_device_index` everywhere a recorder is constructed.
- `make_wakeword_recorder(...)`: builds an `AudioToTextRecorder` (RealtimeSTT) configured with `wakeword_backend="oww"` and an explicit `openwakeword_model_paths=WAKE_WORD_MODEL_PATH` (`hey_jarvis_v0.1.onnx`) — explicit path avoids openWakeWord's default behavior of loading all bundled pretrained models (alexa, hey_mycroft, etc.), which would otherwise also trigger on non-target wake words. `wake_words_sensitivity=0.6`, `wake_word_activation_delay=0.0`, `wake_word_timeout=5.0`.
- `stream_transcribe_with_wakeword(stop_event, language=None, output=None, debug_log=None, max_partial_interval_s=5.0)`: replaces the prior `stream_transcribe` loop. Gates transcription behind wake-word detection via `on_wakeword_detected` callback (`on_wake()`), runs `recorder.text(on_final)` on a background daemon thread (`transcription_loop`), and re-arms wake-word listening after each utterance instead of exiting.
  - `on_realtime_update(text)`: emits partial transcript only when `text_changed` or `interval_elapsed >= max_partial_interval_s` (5s), so live feedback doesn't go silent on long pauses without new words.
  - `on_recorded_chunk(chunk)` / `is_capturing` flag: accumulates raw PCM chunks into `utterance_chunks` only while actively recording an utterance; uses a local `is_capturing` bool rather than reading `recorder.is_recording` because the chunk callback can fire on the recording thread before the `AudioToTextRecorder` constructor returns and binds `recorder`, which previously raced and raised `NameError`.
  - `on_final(text)`: on utterance end, concatenates `utterance_chunks` into an `int16` numpy array and runs the clipping check before printing the finalized utterance and debug-log entry (`debug_log.append(dict(idx=..., elapsed_s=..., text=...))`).
- `clipped_sample_ratio(audio_int16)`: returns fraction of int16 PCM samples at or beyond `INT16_CLIP_VALUE = 32767`. `stream_transcribe_with_wakeword` warns (does not reject) when ratio exceeds `CLIP_RATIO_WARN_THRESHOLD = 0.001` (0.1%), since Whisper transcribes clipped/hot-gain audio fluently but confidently wrong.
- Dataset loader and evaluation routine for speech-to-text accuracy testing.
- Real-time speech-to-text streaming using faster-whisper.

### Fixed
- Pin `onnxruntime` to 1.18.1; newer versions silently broke openWakeWord's ONNX models (loaded inside `make_wakeword_recorder`), leaving wake word scores stuck near zero even on clear speech.
- `stream_transcribe_with_wakeword` re-arms wake-word listening (`emit(f"listening for wake word ({WAKE_WORD})...")`) after each utterance in `on_final`, instead of the recorder going quiet after one detection.
