"""Offline Speech-To-Text Transcriber using faster-whisper (CTranslate2)."""

import io
import logging
from typing import Any, Callable, Optional, Union
import numpy as np

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult, AudioTranscriberPort
from extractoryt.asr.decoder import decode_audio_to_pcm

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(AudioTranscriberPort):
    """
    Offline Speech-To-Text transcriber using faster-whisper (CTranslate2 backend).
    Provides rapid local inference with customizable compute type and device.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        num_threads: int = 4,
        default_language: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_threads = num_threads
        self.default_language = default_language
        self._model: Optional[Any] = None

    def _init_model(
        self, progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Lazily loads the faster-whisper model."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install extractoryt[faster-whisper] or pip install faster-whisper"
            ) from exc

        if progress_callback:
            progress_callback(10.0, f"Loading faster-whisper model '{self.model_size}'...")

        # Determine safe compute_type if default
        compute = self.compute_type
        if compute == "default":
            compute = "int8" if self.device == "cpu" or self.device == "auto" else "float16"

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute,
            cpu_threads=self.num_threads,
        )

    def transcribe(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        language: Optional[str] = None,
    ) -> AudioTranscriptResult:
        """Transcribes input audio using faster-whisper."""
        self._init_model(progress_callback)
        assert self._model is not None

        if isinstance(audio_input, (str, bytes)):
            if progress_callback:
                progress_callback(15.0, "Decoding audio stream...")
            samples, duration = decode_audio_to_pcm(audio_input, target_sr=16000)
        else:
            samples = audio_input.astype(np.float32)
            duration = float(len(samples)) / 16000.0

        if len(samples) == 0:
            return AudioTranscriptResult(
                segments=[],
                language=language or self.default_language or "en",
                duration_seconds=0.0,
                full_text="",
            )

        if progress_callback:
            progress_callback(25.0, "Transcribing audio segments with faster-whisper...")

        lang = language or self.default_language
        segments_gen, info = self._model.transcribe(
            samples,
            language=lang,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        audio_segments: list[AudioSegment] = []
        for segment in segments_gen:
            text = segment.text.strip()
            if text:
                audio_segments.append(
                    AudioSegment(
                        start_seconds=float(segment.start),
                        end_seconds=float(segment.end),
                        text=text,
                    )
                )

        if progress_callback:
            progress_callback(100.0, "Transcription completed.")

        full_text = " ".join(s.text for s in audio_segments)
        detected_language = getattr(info, "language", lang or "en")

        return AudioTranscriptResult(
            segments=audio_segments,
            language=detected_language,
            duration_seconds=duration,
            full_text=full_text,
        )
