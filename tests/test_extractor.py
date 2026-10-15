"""Tests for core YouTube extraction orchestrator."""

from unittest.mock import MagicMock, patch
import pytest

from extractoryt.core import YouTubeExtractor
from extractoryt.exceptions import InvalidYouTubeURLError, TranscriptNotFoundError
from extractoryt.models import TranscriptSnippet, VideoMetadata


@pytest.fixture
def extractor() -> YouTubeExtractor:
    return YouTubeExtractor()


def test_invalid_url_raises_exception(extractor: YouTubeExtractor) -> None:
    with pytest.raises(InvalidYouTubeURLError):
        extractor.extract("https://vimeo.com/123456")


def test_fetch_metadata_success(extractor: YouTubeExtractor) -> None:
    mock_payload = {
        "title": "Clean Architecture in Practice",
        "author_name": "Senior Architect",
        "author_url": "https://www.youtube.com/@seniorarch",
        "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
    }
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        meta = extractor.fetch_metadata("dQw4w9WgXcQ")
        assert meta.title == "Clean Architecture in Practice"
        assert meta.author_name == "Senior Architect"
        assert meta.video_id == "dQw4w9WgXcQ"
        assert meta.video_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_fetch_metadata_fallback_on_network_error(extractor: YouTubeExtractor) -> None:
    with patch("httpx.Client.get", side_effect=Exception("Network error")):
        meta = extractor.fetch_metadata("dQw4w9WgXcQ")
        assert meta.video_id == "dQw4w9WgXcQ"
        assert meta.title == "YouTube Video (dQw4w9WgXcQ)"
        assert meta.author_name == "YouTube"


def test_extract_native_success(extractor: YouTubeExtractor) -> None:
    mock_snippets = [
        TranscriptSnippet(start=0.0, duration=2.0, text="First sentence."),
        TranscriptSnippet(start=2.5, duration=3.0, text="Second sentence."),
    ]
    mock_meta = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Test Video",
        author_name="Test Author",
    )

    with patch.object(extractor, "fetch_metadata", return_value=mock_meta):
        with patch.object(
            extractor,
            "fetch_native_transcript",
            return_value=(mock_snippets, "en", False),
        ):
            res = extractor.extract("dQw4w9WgXcQ")

            assert res.metadata.video_id == "dQw4w9WgXcQ"
            assert res.metadata.title == "Test Video"
            assert len(res.snippets) == 2
            assert res.source == "native_manual"
            assert res.language == "en"
            assert res.is_generated is False
            assert "First sentence. Second sentence." in res.full_text


def test_extract_native_disabled_raises_when_no_fallback(extractor: YouTubeExtractor) -> None:
    mock_meta = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Test Video",
        author_name="Test Author",
    )

    with patch.object(extractor, "fetch_metadata", return_value=mock_meta):
        with patch.object(
            extractor,
            "fetch_native_transcript",
            side_effect=TranscriptNotFoundError("Subtitles disabled"),
        ):
            with pytest.raises(TranscriptNotFoundError):
                extractor.extract("dQw4w9WgXcQ", fallback_to_asr=False)
