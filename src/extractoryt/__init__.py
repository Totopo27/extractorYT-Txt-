"""extractorYT-Txt: High-fidelity YouTube transcript & audio-to-text extraction pipeline."""

from extractoryt.cleaner import clean_sound_artifacts
from extractoryt.core import YouTubeExtractor
from extractoryt.exceptions import (
    ASRTranscriptionError,
    ExtractionError,
    InvalidYouTubeURLError,
    TranscriptNotFoundError,
)
from extractoryt.formatters import (
    JsonFormatter,
    MarkdownFormatter,
    PlainTextFormatter,
    SrtFormatter,
    VttFormatter,
    format_timestamp,
)
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoMetadata

__version__ = "0.1.0"

__all__ = [
    "YouTubeExtractor",
    "clean_sound_artifacts",
    "ExtractionResult",
    "TranscriptSnippet",
    "VideoMetadata",
    "MarkdownFormatter",
    "PlainTextFormatter",
    "JsonFormatter",
    "SrtFormatter",
    "VttFormatter",
    "format_timestamp",
    "ExtractionError",
    "InvalidYouTubeURLError",
    "TranscriptNotFoundError",
    "ASRTranscriptionError",
    "__version__",
]
