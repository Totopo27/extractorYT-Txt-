"""Custom exceptions for extractorYT-Txt."""


class ExtractionError(Exception):
    """Base exception for all YouTube transcript extraction errors."""

    pass


class InvalidYouTubeURLError(ExtractionError, ValueError):
    """Raised when an input string or URL cannot be resolved to a valid YouTube video ID."""

    pass


class TranscriptNotFoundError(ExtractionError, ValueError):
    """Raised when native subtitles/transcripts are disabled or unavailable for a video."""

    pass


class ASRTranscriptionError(ExtractionError, RuntimeError):
    """Raised when offline ASR speech-to-text fallback fails."""

    pass
