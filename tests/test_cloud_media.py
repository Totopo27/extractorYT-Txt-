"""Tests for Google Drive and OneDrive cloud media extraction and authentication."""

import os
from unittest.mock import MagicMock, patch

import pytest

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult
from extractoryt.cli import create_parser, main
from extractoryt.core import CloudMediaExtractor, MediaExtractor, YouTubeExtractor
from extractoryt.exceptions import ExtractionError
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoMetadata


class DummyTranscriber:
    """Mock ASR transcriber for testing."""

    def __init__(self, segments=None, language="en", duration_seconds=5.0):
        self.segments = segments or [
            AudioSegment(start_seconds=0.0, end_seconds=5.0, text="Cloud transcription test [Música]")
        ]
        self.language = language
        self.duration_seconds = duration_seconds

    def transcribe(self, audio_input, progress_callback=None, language=None):
        return AudioTranscriptResult(
            segments=self.segments,
            language=self.language,
            duration_seconds=self.duration_seconds,
            full_text="Cloud transcription test",
        )


# =====================================================================
# URL Detection & Service Identification Tests
# =====================================================================


@pytest.mark.parametrize(
    "url, expected_service",
    [
        ("https://drive.google.com/file/d/12345/view", "gdrive"),
        ("https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view?usp=sharing", "gdrive"),
        ("https://drive.google.com/open?id=12345abcde", "gdrive"),
        ("https://drive.google.com/open?usp=sharing&id=12345abcde", "gdrive"),
        ("https://drive.google.com/uc?id=12345abcde", "gdrive"),
        ("https://drive.google.com/uc?export=download&id=12345abcde", "gdrive"),
        ("https://docs.google.com/file/d/12345abcde", "gdrive"),
        ("drive.google.com/file/d/12345", "gdrive"),
        ("https://drive.google.com/u/0/file/d/12345/view", "gdrive"),
        ("https://drive.google.com/drive/folders/12345", "gdrive"),
        ("https://1drv.ms/v/s!12345", "onedrive"),
        ("https://1drv.ms/u/s!12345", "onedrive"),
        ("1drv.ms/v/s!12345", "onedrive"),
        ("http://1drv.ms/v/s!12345", "onedrive"),
        ("https://onedrive.live.com/?id=12345", "onedrive"),
        ("https://onedrive.live.com/view.aspx?resid=12345", "onedrive"),
        ("onedrive.live.com/?id=12345", "onedrive"),
        ("https://tenant.sharepoint.com/:v:/s/site/12345", "onedrive"),
        ("https://mycompany-my.sharepoint.com/personal/user/doc.mp4", "onedrive"),
        ("tenant.sharepoint.com/:v:/s/12345", "onedrive"),
    ],
)
def test_cloud_url_detection_and_service(url: str, expected_service: str):
    extractor = CloudMediaExtractor()
    assert extractor.is_cloud_url(url) is True
    assert extractor.detect_service(url) == expected_service
    # Also verify static method usage
    assert CloudMediaExtractor.is_cloud_url(url) is True
    assert CloudMediaExtractor.detect_service(url) == expected_service


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
        "https://example.com/audio.mp3",
        "https://vimeo.com/123456",
        "",
        "   ",
    ],
)
def test_non_cloud_urls(url: str):
    assert CloudMediaExtractor.is_cloud_url(url) is False


# =====================================================================
# MediaExtractor Dispatcher Tests
# =====================================================================


def test_media_extractor_dispatch_cloud_gdrive():
    mock_cloud = MagicMock(spec=CloudMediaExtractor)
    mock_cloud.is_cloud_url.return_value = True
    mock_cloud.extract.return_value = MagicMock(spec=ExtractionResult)

    mock_yt = MagicMock(spec=YouTubeExtractor)
    mock_yt.is_youtube_url.return_value = False

    dispatcher = MediaExtractor(youtube_extractor=mock_yt, cloud_extractor=mock_cloud)
    gdrive_url = "https://drive.google.com/file/d/12345/view"

    assert dispatcher.is_cloud_url(gdrive_url) is True
    result = dispatcher.extract(
        gdrive_url,
        cookie_file="/path/to/cookies.txt",
        cookies_from_browser="chrome",
    )

    mock_cloud.extract.assert_called_once()
    _, kwargs = mock_cloud.extract.call_args
    assert kwargs["url"] == gdrive_url
    assert kwargs["cookie_file"] == "/path/to/cookies.txt"
    assert kwargs["cookies_from_browser"] == "chrome"
    mock_yt.extract.assert_not_called()


def test_media_extractor_dispatch_cloud_onedrive():
    mock_cloud = MagicMock(spec=CloudMediaExtractor)
    mock_cloud.is_cloud_url.return_value = True
    mock_cloud.extract.return_value = MagicMock(spec=ExtractionResult)

    dispatcher = MediaExtractor(cloud_extractor=mock_cloud)
    onedrive_url = "https://1drv.ms/v/s!12345"

    result = dispatcher.extract(onedrive_url)
    mock_cloud.extract.assert_called_once()


def test_media_extractor_dispatch_youtube_with_cookies():
    mock_yt = MagicMock(spec=YouTubeExtractor)
    mock_yt.is_youtube_url.return_value = True
    mock_yt.extract.return_value = MagicMock(spec=ExtractionResult)

    dispatcher = MediaExtractor(youtube_extractor=mock_yt)
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    dispatcher.extract(
        yt_url,
        cookie_file="cookies.txt",
        cookies_from_browser="firefox",
    )
    mock_yt.extract.assert_called_once()
    _, kwargs = mock_yt.extract.call_args
    assert kwargs["cookie_file"] == "cookies.txt"
    assert kwargs["cookies_from_browser"] == "firefox"


def test_media_extractor_unsupported_source_message():
    dispatcher = MediaExtractor()
    with pytest.raises(ValueError, match="Unsupported media source"):
        dispatcher.extract("https://unknown-service.com/video")


# =====================================================================
# CloudMediaExtractor Extraction & yt-dlp Options Tests
# =====================================================================


def test_cloud_extractor_extract_gdrive_success():
    transcriber = DummyTranscriber()
    extractor = CloudMediaExtractor(audio_transcriber=transcriber)
    test_url = "https://drive.google.com/file/d/12345abcde/view"

    captured_opts = {}

    def mock_youtube_dl(opts):
        captured_opts.update(opts)
        ydl_instance = MagicMock()

        def mock_extract_info(url, download=True):
            return {
                "id": "12345abcde",
                "title": "Google Drive Lecture",
                "uploader": "Professor Smith",
                "duration": 42.0,
                "thumbnail": "https://example.com/thumb.jpg",
                "webpage_url": url,
            }

        def mock_prepare_filename(info):
            out_path = os.path.join(os.path.dirname(opts["outtmpl"]), "12345abcde.m4a")
            with open(out_path, "wb") as f:
                f.write(b"RIFF_FAKE_AUDIO_DATA")
            return out_path

        ydl_instance.extract_info = mock_extract_info
        ydl_instance.prepare_filename = mock_prepare_filename
        ydl_instance.__enter__ = MagicMock(return_value=ydl_instance)
        ydl_instance.__exit__ = MagicMock(return_value=False)
        return ydl_instance

    with patch("yt_dlp.YoutubeDL", side_effect=mock_youtube_dl):
        result = extractor.extract(
            url=test_url,
            asr_engine="sherpa",
            cookie_file="test_cookies.txt",
            cookies_from_browser="edge",
            clean_noise=True,
        )

    assert isinstance(result, ExtractionResult)
    assert result.source == "gdrive_sherpa"
    assert result.metadata.title == "Google Drive Lecture"
    assert result.metadata.author_name == "Professor Smith"
    assert result.metadata.duration_seconds == 42.0
    assert result.metadata.video_url == test_url
    assert len(result.snippets) == 1
    assert result.snippets[0].text == "Cloud transcription test"

    # Verify cookie configuration passed to yt-dlp
    assert captured_opts.get("cookiefile") == "test_cookies.txt"
    assert captured_opts.get("cookiesfrombrowser") == ("edge", None, None, None)
    assert captured_opts.get("format") == "bestaudio/best"


def test_cloud_extractor_extract_onedrive_success():
    transcriber = DummyTranscriber()
    extractor = CloudMediaExtractor(audio_transcriber=transcriber)
    test_url = "https://1drv.ms/v/s!12345"

    def mock_youtube_dl(opts):
        ydl_instance = MagicMock()

        def mock_extract_info(url, download=True):
            return {
                "id": "onedrive_id_999",
                "title": "Team Sync Video",
                "uploader": "Office 365",
                "duration": 60.0,
                "webpage_url": url,
            }

        def mock_prepare_filename(info):
            out_path = os.path.join(os.path.dirname(opts["outtmpl"]), "audio.m4a")
            with open(out_path, "wb") as f:
                f.write(b"RIFF_FAKE_AUDIO_DATA")
            return out_path

        ydl_instance.extract_info = mock_extract_info
        ydl_instance.prepare_filename = mock_prepare_filename
        ydl_instance.__enter__ = MagicMock(return_value=ydl_instance)
        ydl_instance.__exit__ = MagicMock(return_value=False)
        return ydl_instance

    with patch("yt_dlp.YoutubeDL", side_effect=mock_youtube_dl):
        result = extractor.extract(
            url=test_url,
            asr_engine="faster-whisper",
        )

    assert result.source == "onedrive_faster-whisper"
    assert result.metadata.title == "Team Sync Video"
    assert result.metadata.author_name == "Office 365"


def test_cloud_extractor_invalid_url_raises_error():
    extractor = CloudMediaExtractor()
    with pytest.raises(ValueError, match="Invalid cloud media URL"):
        extractor.extract("https://not-cloud.com/file")


def test_cloud_extractor_download_failure_raises_extraction_error():
    extractor = CloudMediaExtractor()
    with patch("yt_dlp.YoutubeDL") as mock_ydl:
        instance = MagicMock()
        instance.extract_info.side_effect = Exception("yt-dlp network timeout")
        instance.__enter__.return_value = instance
        instance.__exit__.return_value = False
        mock_ydl.return_value = instance

        with pytest.raises(ExtractionError, match="Failed to download cloud audio stream"):
            extractor.download_audio_stream("https://drive.google.com/file/d/12345/view")


# =====================================================================
# YouTubeExtractor Cookies Tests
# =====================================================================


def test_youtube_extractor_cookies_passed_to_ytdlp():
    extractor = YouTubeExtractor()
    captured_opts = {}

    def mock_youtube_dl(opts):
        captured_opts.update(opts)
        ydl_instance = MagicMock()
        ydl_instance.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
        ydl_instance.prepare_filename.side_effect = lambda info: os.path.join(
            os.path.dirname(opts["outtmpl"]), "test.m4a"
        )

        out_path = os.path.join(os.path.dirname(opts["outtmpl"]), "test.m4a")
        with open(out_path, "wb") as f:
            f.write(b"AUDIO_DATA")

        ydl_instance.__enter__ = MagicMock(return_value=ydl_instance)
        ydl_instance.__exit__ = MagicMock(return_value=False)
        return ydl_instance

    with patch("yt_dlp.YoutubeDL", side_effect=mock_youtube_dl):
        extractor.download_audio_stream(
            "dQw4w9WgXcQ",
            cookie_file="/path/to/yt_cookies.txt",
            cookies_from_browser="brave",
        )

    assert captured_opts.get("cookiefile") == "/path/to/yt_cookies.txt"
    assert captured_opts.get("cookiesfrombrowser") == ("brave", None, None, None)


# =====================================================================
# CLI Tests for Cookies & Cloud Arguments
# =====================================================================


def test_cli_parser_cookies_arguments():
    parser = create_parser()
    args = parser.parse_args([
        "https://drive.google.com/file/d/12345/view",
        "--cookies",
        "cookies.txt",
        "--cookies-from-browser",
        "chrome",
    ])
    assert args.source == "https://drive.google.com/file/d/12345/view"
    assert args.cookies == "cookies.txt"
    assert args.cookies_from_browser == "chrome"


def test_cli_main_passes_cookies_to_extractor(capsys):
    mock_result = ExtractionResult(
        metadata=VideoMetadata(
            video_id="mock_id",
            title="Mock Cloud Media",
            author_name="Cloud Author",
            video_url="https://drive.google.com/file/d/12345/view",
            duration_seconds=10.0,
        ),
        snippets=[
            TranscriptSnippet(start=0.0, duration=5.0, text="Transcribed text"),
        ],
        language="en",
        source="gdrive_sherpa",
    )

    with patch("extractoryt.cli.MediaExtractor") as mock_extractor_cls:
        mock_instance = MagicMock()
        mock_instance.is_cloud_url.return_value = True
        mock_instance.extract.return_value = mock_result
        mock_extractor_cls.return_value = mock_instance

        ret = main([
            "https://drive.google.com/file/d/12345/view",
            "--cookies",
            "my_cookies.txt",
            "--cookies-from-browser",
            "firefox",
            "-q",
        ])

        assert ret == 0
        mock_instance.extract.assert_called_once()
        _, kwargs = mock_instance.extract.call_args
        assert kwargs["cookie_file"] == "my_cookies.txt"
        assert kwargs["cookies_from_browser"] == "firefox"
