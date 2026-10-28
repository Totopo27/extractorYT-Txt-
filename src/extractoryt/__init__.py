"""extractorYT-Txt: High-fidelity YouTube transcript & audio-to-text extraction pipeline."""

from extractoryt.cleaner import clean_sound_artifacts
from extractoryt.core import (
    CloudMediaExtractor,
    LocalMediaExtractor,
    MediaExtractor,
    YouTubeExtractor,
    YouTubeTranscriptExtractor,
)
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
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoChapter, VideoMetadata

__version__ = "0.1.0"

__all__ = [
    "YouTubeExtractor",
    "YouTubeTranscriptExtractor",
    "LocalMediaExtractor",
    "CloudMediaExtractor",
    "MediaExtractor",
    "clean_sound_artifacts",
    "ExtractionResult",
    "TranscriptSnippet",
    "VideoChapter",
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
