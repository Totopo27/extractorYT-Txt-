import os
import tempfile
import wave
from unittest.mock import MagicMock, patch

import pytest

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult
from extractoryt.asr.faster_whisper import FasterWhisperTranscriber
from extractoryt.cli import create_parser, main
from extractoryt.core import LocalMediaExtractor, MediaExtractor, YouTubeExtractor
from extractoryt.models import ExtractionResult


@pytest.fixture
def dummy_wav_file():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)

    yield path

    if os.path.exists(path):
        os.remove(path)


class DummyTranscriber:
    def __init__(self, segments=None, language="en", duration_seconds=1.0):
        self.segments = segments or [
            AudioSegment(start_seconds=0.0, end_seconds=1.0, text="Test audio [Música]")
        ]
        self.language = language
        self.duration_seconds = duration_seconds

    def transcribe(self, audio_input, progress_callback=None, language=None):
        return AudioTranscriptResult(
            segments=self.segments,
            language=self.language,
            duration_seconds=self.duration_seconds,
            full_text="Test audio",
        )


def test_local_media_extractor_metadata(dummy_wav_file):
    extractor = LocalMediaExtractor()
    metadata, fmt = extractor.read_metadata(dummy_wav_file)

    assert metadata.video_id == os.path.basename(dummy_wav_file)
    assert metadata.title == os.path.splitext(os.path.basename(dummy_wav_file))[0]
    assert metadata.author_name == "Local Media"
    assert metadata.video_url == str(dummy_wav_file)
    assert metadata.duration_seconds == pytest.approx(1.0, rel=1e-2)
    assert fmt == "wav"


def test_local_media_extractor_extract(dummy_wav_file):
    transcriber = DummyTranscriber()
    extractor = LocalMediaExtractor(audio_transcriber=transcriber)

    result = extractor.extract(dummy_wav_file, clean_noise=True)

    assert isinstance(result, ExtractionResult)
    assert result.source == "local_sherpa"
    assert result.is_generated is True
    assert len(result.snippets) == 1
    assert result.snippets[0].text == "Test audio"


def test_media_extractor_dispatch_local(dummy_wav_file):
    transcriber = DummyTranscriber()
    local_extractor = LocalMediaExtractor(audio_transcriber=transcriber)
    dispatcher = MediaExtractor(local_extractor=local_extractor)

    assert dispatcher.is_local_file(dummy_wav_file) is True
    assert dispatcher.is_youtube_url(dummy_wav_file) is False

    result = dispatcher.extract(dummy_wav_file)
    assert result.source == "local_sherpa"
    assert result.metadata.author_name == "Local Media"


def test_media_extractor_dispatch_youtube():
    mock_yt = MagicMock()
    mock_yt.is_youtube_url.return_value = True
    mock_yt.extract.return_value = MagicMock(spec=ExtractionResult)

    dispatcher = MediaExtractor(youtube_extractor=mock_yt)
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    assert dispatcher.is_youtube_url(yt_url) is True
    result = dispatcher.extract(yt_url)
    mock_yt.extract.assert_called_once()


def test_media_extractor_invalid_source():
    dispatcher = MediaExtractor()
    with pytest.raises(ValueError, match="Unsupported media source"):
        dispatcher.extract("not_a_real_file_or_youtube_url.xyz")


def test_faster_whisper_device_detection_cuda():
    with patch("ctranslate2.get_cuda_device_count", return_value=1):
        with patch("faster_whisper.WhisperModel") as mock_model:
            transcriber = FasterWhisperTranscriber(model_size="tiny", device="auto")
            transcriber._init_model()
            mock_model.assert_called_once()
            _, kwargs = mock_model.call_args
            assert kwargs["device"] == "cuda"
            assert kwargs["compute_type"] == "float16"


def test_faster_whisper_device_detection_cpu():
    with patch("ctranslate2.get_cuda_device_count", return_value=0):
        with patch("faster_whisper.WhisperModel") as mock_model:
            transcriber = FasterWhisperTranscriber(model_size="tiny", device="auto")
            transcriber._init_model()
            mock_model.assert_called_once()
            _, kwargs = mock_model.call_args
            assert kwargs["device"] == "cpu"
            assert kwargs["compute_type"] == "int8"


def test_faster_whisper_cuda_runtime_error_fallback_to_cpu():
    from unittest.mock import MagicMock
    import numpy as np

    transcriber = FasterWhisperTranscriber(model_size="tiny", device="cuda")
    mock_cuda_model = MagicMock()
    mock_cuda_model.transcribe.side_effect = RuntimeError("Library cublas64_12.dll is not found")

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "Fallback text"

    mock_cpu_model = MagicMock()
    mock_cpu_model.transcribe.return_value = ([mock_segment], MagicMock(language="es"))

    transcriber._model = mock_cuda_model

    with patch("faster_whisper.WhisperModel", return_value=mock_cpu_model) as mock_whisper_cls:
        res = transcriber.transcribe(np.zeros(16000, dtype=np.float32), language="es")
        mock_whisper_cls.assert_called_once()
        _, kwargs = mock_whisper_cls.call_args
        assert kwargs["device"] == "cpu"
        assert len(res.segments) == 1
        assert res.segments[0].text == "Fallback text"


def test_cli_parser_source_and_device():
    parser = create_parser()
    args = parser.parse_args(["sample.mp4", "--device", "cuda", "--asr-engine", "faster-whisper"])
    assert args.source == "sample.mp4"
    assert args.device == "cuda"
    assert args.asr_engine == "faster-whisper"
