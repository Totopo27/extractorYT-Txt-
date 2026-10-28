"""Comprehensive tests for native chapters support across models, formatters, core, and CLI."""

from unittest.mock import MagicMock, patch

import pytest

from extractoryt.cli import create_parser, main
from extractoryt.core import (
    CloudMediaExtractor,
    MediaExtractor,
    YouTubeExtractor,
    YouTubeTranscriptExtractor,
)
from extractoryt.formatters import JsonFormatter, MarkdownFormatter
from extractoryt.models import (
    ExtractionResult,
    TranscriptSnippet,
    VideoChapter,
    VideoMetadata,
)


class TestVideoChapterModel:
    """Tests for VideoChapter model creation and serialization."""

    def test_video_chapter_creation(self) -> None:
        chapter = VideoChapter(title="Introducción", start=0.0, end=120.5)
        assert chapter.title == "Introducción"
        assert chapter.start == 0.0
        assert chapter.end == 120.5

    def test_video_chapter_to_dict(self) -> None:
        chapter = VideoChapter(title="Deep Dive", start=12.3456, end=78.9012)
        serialized = chapter.to_dict()
        assert serialized == {
            "title": "Deep Dive",
            "start": 12.346,
            "end": 78.901,
        }

    def test_video_metadata_with_chapters(self) -> None:
        chapters = [
            VideoChapter(title="Intro", start=0.0, end=60.0),
            VideoChapter(title="Conclusion", start=60.0, end=120.0),
        ]
        meta = VideoMetadata(
            video_id="dQw4w9WgXcQ",
            title="Rick Roll",
            author_name="Rick Astley",
            chapters=chapters,
        )
        assert len(meta.chapters) == 2
        data = meta.to_dict()
        assert "chapters" in data
        assert len(data["chapters"]) == 2
        assert data["chapters"][0]["title"] == "Intro"
        assert data["chapters"][0]["start"] == 0.0
        assert data["chapters"][0]["end"] == 60.0
        assert data["chapters"][1]["title"] == "Conclusion"

    def test_video_metadata_default_empty_chapters(self) -> None:
        meta = VideoMetadata(
            video_id="dQw4w9WgXcQ",
            title="Rick Roll",
            author_name="Rick Astley",
        )
        assert meta.chapters == []
        assert meta.to_dict()["chapters"] == []


class TestMarkdownFormatterChapters:
    """Tests for MarkdownFormatter rendering chapters TOC and chapter-aligned headings."""

    def test_markdown_with_chapters_rendering(self) -> None:
        chapters = [
            VideoChapter(title="Introducción", start=0.0, end=270.0),
            VideoChapter(title="Configuración del entorno", start=270.0, end=600.0),
        ]
        meta = VideoMetadata(
            video_id="vid123",
            title="Python Tutorial",
            author_name="Dev Channel",
            chapters=chapters,
        )
        snippets = [
            TranscriptSnippet(start=0.0, duration=10.0, text="Bienvenidos al curso."),
            TranscriptSnippet(start=15.0, duration=10.0, text="Hoy veremos Python."),
            TranscriptSnippet(start=270.0, duration=12.0, text="Comenzamos configurando el IDE."),
            TranscriptSnippet(start=300.0, duration=10.0, text="Instalamos dependencias."),
        ]
        result = ExtractionResult(metadata=meta, snippets=snippets, language="es")
        formatter = MarkdownFormatter()
        output = formatter.format(result)

        # Check TOC section
        assert "## Chapters" in output
        assert "- **[00:00]** Introducción" in output
        assert "- **[04:30]** Configuración del entorno" in output

        # Check body chapter headings
        assert "### [00:00] Introducción" in output
        assert "### [04:30] Configuración del entorno" in output

        # Ensure generic timestamp headers are NOT present when chapters exist
        assert "Timestamp 00:00" not in output
        assert "Timestamp 04:30" not in output

        # Check paragraph text is included
        assert "Bienvenidos al curso." in output
        assert "Comenzamos configurando el IDE." in output

    def test_markdown_fallback_when_no_chapters(self) -> None:
        meta = VideoMetadata(
            video_id="vid123",
            title="Python Tutorial",
            author_name="Dev Channel",
            chapters=[],
        )
        snippets = [
            TranscriptSnippet(start=0.0, duration=10.0, text="Intro without chapters."),
            TranscriptSnippet(start=305.0, duration=10.0, text="After five minutes."),
        ]
        result = ExtractionResult(metadata=meta, snippets=snippets, language="en")
        formatter = MarkdownFormatter(heading_interval_seconds=300.0)
        output = formatter.format(result)

        # TOC should NOT be present
        assert "## Chapters" not in output

        # Generic timestamps SHOULD be present
        assert "### [00:00] Timestamp 00:00" in output
        assert "### [05:05] Timestamp 05:05" in output

    def test_markdown_with_chapters_but_no_snippets(self) -> None:
        chapters = [VideoChapter(title="Solo Capitulo", start=0.0, end=100.0)]
        meta = VideoMetadata(
            video_id="vid123",
            title="Empty Video",
            author_name="Empty Channel",
            chapters=chapters,
        )
        result = ExtractionResult(metadata=meta, snippets=[], language="es")
        formatter = MarkdownFormatter()
        output = formatter.format(result)

        assert "## Chapters" in output
        assert "- **[00:00]** Solo Capitulo" in output
        assert "*(No transcript snippets found)*" in output

    def test_json_formatter_includes_chapters(self) -> None:
        chapters = [VideoChapter(title="Chapter 1", start=0.0, end=50.0)]
        meta = VideoMetadata(
            video_id="vid123",
            title="JSON Test",
            author_name="Channel",
            chapters=chapters,
        )
        result = ExtractionResult(metadata=meta, snippets=[], language="en")
        output = JsonFormatter().format(result)
        assert '"chapters": [' in output
        assert '"title": "Chapter 1"' in output


class TestYouTubeExtractorChapters:
    """Tests for YouTubeExtractor / YouTubeTranscriptExtractor fetch_chapters and extraction."""

    def test_alias_exists(self) -> None:
        assert YouTubeTranscriptExtractor is YouTubeExtractor

    def test_fetch_chapters_success(self) -> None:
        extractor = YouTubeExtractor()
        mock_info = {
            "chapters": [
                {"title": "Intro", "start_time": 0.0, "end_time": 45.0},
                {"title": "Core Concepts", "start_time": 45.0, "end_time": 180.0},
            ]
        }
        with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.extract_info.return_value = mock_info
            mock_ydl_cls.return_value = mock_instance

            chapters = extractor.fetch_chapters("test_video_id")
            assert len(chapters) == 2
            assert chapters[0].title == "Intro"
            assert chapters[0].start == 0.0
            assert chapters[0].end == 45.0
            assert chapters[1].title == "Core Concepts"
            assert chapters[1].start == 45.0
            assert chapters[1].end == 180.0

    def test_fetch_chapters_no_chapters_key(self) -> None:
        extractor = YouTubeExtractor()
        mock_info = {"id": "test_id"}
        with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.extract_info.return_value = mock_info
            mock_ydl_cls.return_value = mock_instance

            chapters = extractor.fetch_chapters("test_video_id")
            assert chapters == []

    def test_fetch_chapters_import_error(self) -> None:
        extractor = YouTubeExtractor()
        with patch.dict("sys.modules", {"yt_dlp": None}):
            chapters = extractor.fetch_chapters("test_video_id")
            assert chapters == []

    def test_fetch_chapters_exception_handled_gracefully(self) -> None:
        extractor = YouTubeExtractor()
        with patch("yt_dlp.YoutubeDL", side_effect=RuntimeError("Network error")):
            chapters = extractor.fetch_chapters("test_video_id")
            assert chapters == []

    def test_extract_calls_fetch_chapters_when_enabled(self) -> None:
        extractor = YouTubeExtractor()
        mock_meta = VideoMetadata(video_id="abc12345678", title="Test", author_name="Author")
        mock_snippets = [TranscriptSnippet(start=0.0, duration=5.0, text="Hi")]
        mock_chapters = [VideoChapter(title="Intro", start=0.0, end=5.0)]

        with patch.object(extractor, "fetch_metadata", return_value=mock_meta), \
             patch.object(extractor, "fetch_native_transcript", return_value=(mock_snippets, "en", False)), \
             patch.object(extractor, "fetch_chapters", return_value=mock_chapters) as mock_fetch:

            result = extractor.extract("https://www.youtube.com/watch?v=abc12345678", extract_chapters=True)
            mock_fetch.assert_called_once_with(
                video_id="abc12345678",
                cookie_file=None,
                cookies_from_browser=None,
            )
            assert result.metadata.chapters == mock_chapters

    def test_extract_skips_chapters_when_disabled(self) -> None:
        extractor = YouTubeExtractor()
        mock_meta = VideoMetadata(video_id="abc12345678", title="Test", author_name="Author")
        mock_snippets = [TranscriptSnippet(start=0.0, duration=5.0, text="Hi")]

        with patch.object(extractor, "fetch_metadata", return_value=mock_meta), \
             patch.object(extractor, "fetch_native_transcript", return_value=(mock_snippets, "en", False)), \
             patch.object(extractor, "fetch_chapters") as mock_fetch:

            result = extractor.extract("https://www.youtube.com/watch?v=abc12345678", extract_chapters=False)
            mock_fetch.assert_not_called()
            assert result.metadata.chapters == []


class TestCloudMediaChapters:
    """Tests for CloudMediaExtractor parsing chapters from entry_info."""

    def test_cloud_media_parses_chapters(self) -> None:
        extractor = CloudMediaExtractor()
        mock_info = {
            "id": "file123",
            "title": "Meeting Recording",
            "uploader": "Corporate",
            "duration": 3600.0,
            "chapters": [
                {"title": "Agenda", "start_time": 0.0, "end_time": 300.0},
                {"title": "Q&A", "start_time": 300.0, "end_time": 600.0},
            ],
        }

        with patch("yt_dlp.YoutubeDL") as mock_ydl_cls, \
             patch("builtins.open"), \
             patch("os.path.isfile", return_value=True):
            mock_instance = MagicMock()
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.extract_info.return_value = mock_info
            mock_instance.prepare_filename.return_value = "dummy.mp3"
            mock_ydl_cls.return_value = mock_instance

            _, metadata = extractor.download_audio_stream("https://drive.google.com/file/d/123/view")
            assert len(metadata.chapters) == 2
            assert metadata.chapters[0].title == "Agenda"
            assert metadata.chapters[0].start == 0.0
            assert metadata.chapters[0].end == 300.0
            assert metadata.chapters[1].title == "Q&A"

    def test_cloud_media_alias_extract_cloud_media(self) -> None:
        extractor = CloudMediaExtractor()
        assert hasattr(extractor, "extract_cloud_media")
        assert extractor.extract_cloud_media == extractor.extract


class TestMediaExtractorDelegation:
    """Tests for MediaExtractor propagating extract_chapters flag."""

    def test_media_extractor_delegates_extract_chapters(self) -> None:
        mock_yt = MagicMock()
        extractor = MediaExtractor(youtube_extractor=mock_yt)
        extractor.extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ", extract_chapters=False)
        mock_yt.extract.assert_called_once()
        assert mock_yt.extract.call_args[1]["extract_chapters"] is False


class TestCliChaptersFlag:
    """Tests for CLI --no-chapters flag and summary display."""

    def test_cli_parser_has_no_chapters(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["https://www.youtube.com/watch?v=abc12345678", "--no-chapters"])
        assert args.no_chapters is True

        args_default = parser.parse_args(["https://www.youtube.com/watch?v=abc12345678"])
        assert args_default.no_chapters is False

    def test_cli_main_passes_extract_chapters(self) -> None:
        mock_meta = VideoMetadata(
            video_id="abc12345678",
            title="Test",
            author_name="Author",
            chapters=[VideoChapter(title="Ch 1", start=0.0, end=10.0)],
        )
        mock_result = ExtractionResult(metadata=mock_meta, snippets=[], language="en")

        with patch("extractoryt.cli.MediaExtractor") as mock_media_extractor_cls:
            mock_instance = MagicMock()
            mock_instance.is_cloud_url.return_value = False
            mock_instance.extract.return_value = mock_result
            mock_media_extractor_cls.return_value = mock_instance

            ret = main(["https://www.youtube.com/watch?v=abc12345678", "--no-chapters", "-q"])
            assert ret == 0
            assert mock_instance.extract.call_args[1]["extract_chapters"] is False
