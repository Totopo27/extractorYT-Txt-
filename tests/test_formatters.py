"""Tests for transcript formatters (Markdown, PlainText, SRT, VTT, JSON)."""

import json
import pytest

from extractoryt.formatters import (
    JsonFormatter,
    MarkdownFormatter,
    PlainTextFormatter,
    SrtFormatter,
    VttFormatter,
    format_timestamp,
)
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoMetadata


@pytest.fixture
def sample_result() -> ExtractionResult:
    metadata = VideoMetadata(
        video_id="test_video_123",
        title="Quantum Mechanics 101",
        author_name="MIT OpenCourseWare",
        author_url="https://www.youtube.com/@mitocw",
        thumbnail_url="https://i.ytimg.com/vi/test_video_123/hqdefault.jpg",
        video_url="https://www.youtube.com/watch?v=test_video_123",
    )
    snippets = [
        TranscriptSnippet(start=0.0, duration=2.5, text="Welcome to quantum mechanics."),
        TranscriptSnippet(start=3.0, duration=3.2, text="Today we will discuss the double slit experiment."),
        TranscriptSnippet(start=7.0, duration=2.8, text="Particles act as waves when not observed."),
        TranscriptSnippet(start=310.0, duration=4.0, text="Now let us move to wave function collapse."),
        TranscriptSnippet(start=315.0, duration=3.5, text="This leads directly to Schrodinger equation."),
    ]
    return ExtractionResult(
        metadata=metadata,
        snippets=snippets,
        language="en",
        is_generated=False,
        source="native_manual",
    )


def test_format_timestamp_standard() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(45) == "00:45"
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(3665) == "01:01:05"


def test_format_timestamp_srt() -> None:
    assert format_timestamp(0, srt_mode=True) == "00:00:00,000"
    assert format_timestamp(75.5, srt_mode=True) == "00:01:15,500"
    assert format_timestamp(3665.123, srt_mode=True) == "01:01:05,123"


def test_format_timestamp_vtt() -> None:
    assert format_timestamp(0, vtt_mode=True) == "00:00:00.000"
    assert format_timestamp(75.5, vtt_mode=True) == "00:01:15.500"
    assert format_timestamp(3665.123, vtt_mode=True) == "01:01:05.123"


def test_markdown_formatter(sample_result: ExtractionResult) -> None:
    formatter = MarkdownFormatter()
    md = formatter.format(sample_result)

    assert "# Quantum Mechanics 101" in md
    assert "MIT OpenCourseWare" in md
    assert "https://www.youtube.com/watch?v=test_video_123" in md
    assert "[00:00]" in md
    assert "Welcome to quantum mechanics." in md
    assert "### [05:10] Timestamp 05:10" in md
    assert "Schrodinger equation" in md


def test_markdown_formatter_empty() -> None:
    metadata = VideoMetadata(video_id="empty", title="Empty", author_name="None")
    empty_result = ExtractionResult(metadata=metadata, snippets=[])
    md = MarkdownFormatter().format(empty_result)
    assert "# Empty" in md
    assert "*(No transcript snippets found)*" in md


def test_plain_text_formatter(sample_result: ExtractionResult) -> None:
    plain_formatter = PlainTextFormatter(include_timestamps=False)
    txt = plain_formatter.format(sample_result)
    assert "Welcome to quantum mechanics." in txt
    assert "Today we will discuss" in txt
    assert "[00:00]" not in txt

    ts_formatter = PlainTextFormatter(include_timestamps=True)
    ts_txt = ts_formatter.format(sample_result)
    assert "[00:00] Welcome to quantum mechanics." in ts_txt
    assert "[05:10] Now let us move to wave function collapse." in ts_txt


def test_srt_formatter(sample_result: ExtractionResult) -> None:
    formatter = SrtFormatter()
    srt = formatter.format(sample_result)

    assert "1\n00:00:00,000 --> 00:00:02,500\nWelcome to quantum mechanics." in srt
    assert "00:05:10,000 --> 00:05:14,000" in srt


def test_vtt_formatter(sample_result: ExtractionResult) -> None:
    formatter = VttFormatter()
    vtt = formatter.format(sample_result)

    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500" in vtt
    assert "Welcome to quantum mechanics." in vtt


def test_json_formatter(sample_result: ExtractionResult) -> None:
    formatter = JsonFormatter()
    json_str = formatter.format(sample_result)
    data = json.loads(json_str)

    assert data["metadata"]["title"] == "Quantum Mechanics 101"
    assert data["metadata"]["video_id"] == "test_video_123"
    assert data["language"] == "en"
    assert data["source"] == "native_manual"
    assert len(data["snippets"]) == 5
    assert data["snippets"][0]["text"] == "Welcome to quantum mechanics."
    assert data["snippets"][0]["start"] == 0.0
