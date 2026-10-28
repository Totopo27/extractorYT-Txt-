"""Factory for instantiating speech recognition / ASR transcribers."""

from typing import Any, Optional
from extractoryt.asr.base import AudioTranscriberPort


def get_transcriber(
    engine: str = "sherpa",
    model_size: str = "small",
    device: str = "auto",
    **kwargs: Any,
) -> AudioTranscriberPort:
    """
    Returns an instance of an AudioTranscriberPort based on engine name.

    Supported engines:
    - 'sherpa' or 'sherpa-onnx' -> SherpaOnnxTranscriber
    - 'faster-whisper' or 'faster_whisper' -> FasterWhisperTranscriber
    """
    normalized_engine = engine.strip().lower().replace("_", "-")

    if normalized_engine in ("sherpa", "sherpa-onnx", "onnx"):
        from extractoryt.asr.sherpa import SherpaOnnxTranscriber

        return SherpaOnnxTranscriber(model_size=model_size, device=device, **kwargs)

    elif normalized_engine in ("faster-whisper", "fasterwhisper", "ctranslate2"):
        from extractoryt.asr.faster_whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber(model_size=model_size, device=device, **kwargs)

    raise ValueError(
        f"Unsupported ASR engine: '{engine}'. Available options are 'sherpa' or 'faster-whisper'."
    )
