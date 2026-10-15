"""Ports / Interfaces for YouTube extraction and audio transcription."""

from typing import Any, Callable, Optional, Protocol, Union
import numpy as np

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult, AudioTranscriberPort
from extractoryt.models import TranscriptSnippet, VideoMetadata


class MetadataProviderPort(Protocol):
    """Protocol for fetching video metadata."""

    def fetch_metadata(self, video_id: str) -> VideoMetadata:
        """Fetch metadata for a given video ID."""
        ...


class TranscriptProviderPort(Protocol):
    """Protocol for fetching native transcripts."""

    def fetch_transcript(
        self,
        video_id: str,
        preferred_languages: Optional[list[str]] = None,
    ) -> tuple[list[TranscriptSnippet], str, bool]:
        """Fetch transcript snippets, returning (snippets, language, is_generated)."""
        ...


__all__ = [
    "AudioSegment",
    "AudioTranscriptResult",
    "AudioTranscriberPort",
    "MetadataProviderPort",
    "TranscriptProviderPort",
]
