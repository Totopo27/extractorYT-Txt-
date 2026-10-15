"""Tests for ASR fallback flow, transcriber protocols, and factory."""

from unittest.mock import MagicMock, patch
import pytest

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult, AudioTranscriberPort
from extractoryt.asr.factory import get_transcriber
from extractoryt.core import YouTubeExtractor
from extractoryt.exceptions import TranscriptNotFoundError
from extractoryt.models import VideoMetadata


class MockTranscriber(AudioTranscriberPort):
    def transcribe(
        self,
        audio_input,
        progress_callback=None,
        language=None,
    ) -> AudioTranscriptResult:
        if progress_callback:
            progress_callback(50.0, "Transcribing in mock...")
        return AudioTranscriptResult(
            segments=[
                AudioSegment(start_seconds=1.0, end_seconds=4.0, text="Transcribed via mock ASR."),
            ],
            language=language or "es",
            duration_seconds=5.0,
            full_text="Transcribed via mock ASR.",
        )


def test_asr_fallback_when_native_fails() -> None:
    mock_transcriber = MockTranscriber()
    extractor = YouTubeExtractor(audio_transcriber=mock_transcriber)

    mock_meta = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Podcast Episode",
        author_name="Podcaster",
    )

    with patch.object(extractor, "fetch_metadata", return_value=mock_meta):
        with patch.object(
            extractor,
            "fetch_native_transcript",
            side_effect=TranscriptNotFoundError("No subtitles found"),
        ):
            with patch.object(
                extractor,
                "download_audio_stream",
                return_value=b"RIFFdummywavebytes",
            ):
                res = extractor.extract(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    fallback_to_asr=True,
                )

                assert res.source == "asr_sherpa"
                assert res.is_generated is True
                assert len(res.snippets) == 1
                assert res.snippets[0].text == "Transcribed via mock ASR."
                assert res.snippets[0].start == 1.0


def test_force_asr_bypasses_native() -> None:
    mock_transcriber = MockTranscriber()
    extractor = YouTubeExtractor(audio_transcriber=mock_transcriber)

    mock_meta = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Podcast Episode",
        author_name="Podcaster",
    )

    with patch.object(extractor, "fetch_metadata", return_value=mock_meta):
        with patch.object(extractor, "fetch_native_transcript") as mock_native:
            with patch.object(
                extractor,
                "download_audio_stream",
                return_value=b"RIFFdummywavebytes",
            ):
                res = extractor.extract("dQw4w9WgXcQ", force_asr=True)

                mock_native.assert_not_called()
                assert res.source == "asr_sherpa"
                assert res.snippets[0].text == "Transcribed via mock ASR."


def test_transcriber_factory_sherpa() -> None:
    transcriber = get_transcriber("sherpa", model_size="tiny")
    assert transcriber.__class__.__name__ == "SherpaOnnxTranscriber"
    assert transcriber.model_size == "tiny"


def test_transcriber_factory_faster_whisper() -> None:
    transcriber = get_transcriber("faster-whisper", model_size="base")
    assert transcriber.__class__.__name__ == "FasterWhisperTranscriber"
    assert transcriber.model_size == "base"


def test_transcriber_factory_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported ASR engine"):
        get_transcriber("invalid-engine")


def test_decode_audio_to_pcm() -> None:
    import io
    import wave
    import numpy as np
    from extractoryt.asr.decoder import decode_audio_to_pcm

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = (np.sin(np.linspace(0, 2 * np.pi * 440, 1600)) * 32767).astype(np.int16)
        wf.writeframes(samples.tobytes())

    pcm, duration = decode_audio_to_pcm(buf.getvalue(), target_sr=16000)
    assert isinstance(pcm, np.ndarray)
    assert pcm.dtype == np.float32
    assert len(pcm) == 1600
    assert pytest.approx(duration, 0.01) == 0.1

