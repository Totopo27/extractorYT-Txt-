"""Data models for transcripts, metadata, and extraction results."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TranscriptSnippet:
    """A granular timestamped transcript entry."""

    start: float
    duration: float
    text: str
    speaker: Optional[str] = None

    @property
    def end(self) -> float:
        """Returns the calculated end timestamp in seconds."""
        return self.start + max(0.0, self.duration)

    def to_dict(self) -> dict[str, Any]:
        """Serializes snippet to dictionary."""
        data: dict[str, Any] = {
            "start": round(self.start, 3),
            "duration": round(self.duration, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        if self.speaker:
            data["speaker"] = self.speaker
        return data


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video retrieved via oEmbed or direct extraction."""

    video_id: str
    title: str
    author_name: str
    author_url: str = ""
    thumbnail_url: str = ""
    video_url: str = ""
    duration_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.video_url and self.video_id:
            self.video_url = f"https://www.youtube.com/watch?v={self.video_id}"
        if not self.thumbnail_url and self.video_id:
            self.thumbnail_url = f"https://i.ytimg.com/vi/{self.video_id}/hqdefault.jpg"

    def to_dict(self) -> dict[str, Any]:
        """Serializes video metadata to dictionary."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "author_name": self.author_name,
            "author_url": self.author_url,
            "thumbnail_url": self.thumbnail_url,
            "video_url": self.video_url,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ExtractionResult:
    """Full container for transcript extraction results."""

    metadata: VideoMetadata
    snippets: list[TranscriptSnippet] = field(default_factory=list)
    language: str = "en"
    is_generated: bool = False
    source: str = "native"  # "native_manual", "native_generated", "asr_sherpa", "asr_faster_whisper"

    @property
    def duration_seconds(self) -> float:
        """Calculates total duration from metadata or final snippet."""
        if self.metadata.duration_seconds is not None and self.metadata.duration_seconds > 0:
            return self.metadata.duration_seconds
        if self.snippets:
            last = self.snippets[-1]
            return last.end
        return 0.0

    @property
    def full_text(self) -> str:
        """Concatenates all snippets into continuous plain text."""
        return " ".join(s.text.strip() for s in self.snippets if s.text.strip())

    def to_dict(self) -> dict[str, Any]:
        """Serializes complete extraction result to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "language": self.language,
            "is_generated": self.is_generated,
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 2),
            "snippet_count": len(self.snippets),
            "snippets": [s.to_dict() for s in self.snippets],
        }
