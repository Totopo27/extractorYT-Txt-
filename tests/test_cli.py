"""Tests for CLI interface."""

import json
from unittest.mock import MagicMock, patch
import pytest

from extractoryt.cli import create_parser, main
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoMetadata


@pytest.fixture
def dummy_result() -> ExtractionResult:
    return ExtractionResult(
        metadata=VideoMetadata(
            video_id="dQw4w9WgXcQ",
            title="Never Gonna Give You Up",
            author_name="Rick Astley",
        ),
        snippets=[
            TranscriptSnippet(start=0.0, duration=2.0, text="Never gonna give you up."),
            TranscriptSnippet(start=2.5, duration=2.0, text="Never gonna let you down."),
        ],
        language="en",
        is_generated=False,
        source="native_manual",
    )


def test_cli_parser_defaults() -> None:
    parser = create_parser()
    args = parser.parse_args(["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
    assert args.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert args.format == "markdown"
    assert args.output == "-"
    assert args.lang == "es,en"
    assert args.no_fallback is False
    assert args.force_asr is False
    assert args.asr_engine == "sherpa"


def test_cli_stdout_markdown(capsys, dummy_result: ExtractionResult) -> None:
    with patch("extractoryt.cli.YouTubeExtractor.extract", return_value=dummy_result):
        exit_code = main(["dQw4w9WgXcQ", "--quiet"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "# Never Gonna Give You Up" in captured.out
        assert "Never gonna give you up." in captured.out


def test_cli_stdout_json(capsys, dummy_result: ExtractionResult) -> None:
    with patch("extractoryt.cli.YouTubeExtractor.extract", return_value=dummy_result):
        exit_code = main(["dQw4w9WgXcQ", "-f", "json", "-q"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["metadata"]["title"] == "Never Gonna Give You Up"
        assert len(data["snippets"]) == 2


def test_cli_output_file(tmp_path, dummy_result: ExtractionResult) -> None:
    output_file = tmp_path / "transcript.md"
    with patch("extractoryt.cli.YouTubeExtractor.extract", return_value=dummy_result):
        exit_code = main(["dQw4w9WgXcQ", "-o", str(output_file), "-q"])
        assert exit_code == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# Never Gonna Give You Up" in content


def test_cli_invalid_url_error(capsys) -> None:
    exit_code = main(["https://vimeo.com/12345", "-q"])
    assert exit_code != 0
