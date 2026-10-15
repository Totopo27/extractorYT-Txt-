"""Tests for YouTube URL parsing and video ID extraction."""

import pytest
from extractoryt.core import YouTubeExtractor


@pytest.fixture
def extractor() -> YouTubeExtractor:
    return YouTubeExtractor()


def test_is_youtube_url_valid(extractor: YouTubeExtractor) -> None:
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://youtu.be/dQw4w9WgXcQ?t=120",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "youtube.com/watch?v=dQw4w9WgXcQ",
        "m.youtube.com/watch?v=dQw4w9WgXcQ",
        "dQw4w9WgXcQ",  # Raw 11-char ID
    ]
    for url in valid_urls:
        assert extractor.is_youtube_url(url), f"Expected '{url}' to be recognized as YouTube URL"


def test_is_youtube_url_invalid(extractor: YouTubeExtractor) -> None:
    invalid_urls = [
        "",
        "   ",
        "https://example.com/video",
        "https://vimeo.com/12345678",
        "10.1038/nature12345",
        "not_a_valid_id",
        "dQw4w9WgXc",  # 10 chars
        "dQw4w9WgXcQQQ",  # 13 chars
    ]
    for url in invalid_urls:
        assert not extractor.is_youtube_url(url), f"Expected '{url}' to be rejected"


def test_extract_video_id(extractor: YouTubeExtractor) -> None:
    expected = "dQw4w9WgXcQ"
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&t=45s",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ?si=123",
        "dQw4w9WgXcQ",
    ]
    for url in urls:
        assert extractor.extract_video_id(url) == expected, f"Failed for {url}"


def test_extract_video_id_invalid(extractor: YouTubeExtractor) -> None:
    assert extractor.extract_video_id("https://example.com") is None
    assert extractor.extract_video_id("") is None
    assert extractor.extract_video_id("invalid") is None
