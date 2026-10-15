"""ASR (Automatic Speech Recognition) module for offline audio-to-text fallback."""

from extractoryt.asr.base import (
    AudioSegment,
    AudioTranscriptResult,
    AudioTranscriberPort,
)
from extractoryt.asr.factory import get_transcriber

__all__ = [
    "AudioSegment",
    "AudioTranscriptResult",
    "AudioTranscriberPort",
    "get_transcriber",
]
