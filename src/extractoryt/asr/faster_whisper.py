"""Offline Speech-To-Text Transcriber using faster-whisper (CTranslate2)."""

import io
import logging
import os
import sys
from typing import Any, Callable, Optional, Union
import numpy as np

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult, AudioTranscriberPort
from extractoryt.asr.decoder import decode_audio_to_pcm

logger = logging.getLogger(__name__)


def _register_nvidia_dll_directories() -> None:
    """On Windows, registers NVIDIA pip package DLL directories into DLL search path if available."""
    if sys.platform != "win32":
        return
    try:
        import site

        roots = []
        try:
            roots.extend(site.getsitepackages())
        except AttributeError:
            pass
        try:
            roots.append(site.getusersitepackages())
        except AttributeError:
            pass

        for p in roots:
            nvidia_dir = os.path.join(p, "nvidia")
            if os.path.isdir(nvidia_dir):
                for root, _, files in os.walk(nvidia_dir):
                    if any(f.lower().endswith(".dll") for f in files):
                        try:
                            os.add_dll_directory(root)
                        except (AttributeError, OSError):
                            pass
                        os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:
        logger.debug("Could not auto-register NVIDIA DLL directories: %s", exc)


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

        _register_nvidia_dll_directories()

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install extractoryt[faster-whisper] or pip install faster-whisper"
            ) from exc

        if progress_callback:
            progress_callback(10.0, f"Loading faster-whisper model '{self.model_size}'...")

        device = self.device
        compute = self.compute_type

        if device == "auto":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                    compute = "float16"
                else:
                    device = "cpu"
                    compute = "int8"
            except Exception as exc:
                logger.warning(
                    "Could not detect CUDA device count via ctranslate2: %s. Defaulting to CPU.", exc
                )
                device = "cpu"
                compute = "int8"
        elif compute == "default":
            compute = "int8" if device == "cpu" else "float16"

        self._model = WhisperModel(
            self.model_size,
            device=device,
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

        def _run_transcription(model: Any, is_retry: bool = False) -> tuple[list[AudioSegment], Any]:
            segments_gen, info = model.transcribe(
                samples,
                language=lang,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            audio_segs: list[AudioSegment] = []
            last_pct = 25.0
            for segment in segments_gen:
                text = segment.text.strip()
                if text:
                    audio_segs.append(
                        AudioSegment(
                            start_seconds=float(segment.start),
                            end_seconds=float(segment.end),
                            text=text,
                        )
                    )
                if progress_callback and duration > 0:
                    current_pct = 25.0 + (float(segment.end) / duration) * 70.0
                    if current_pct - last_pct >= 5.0 or current_pct >= 94.0:
                        last_pct = current_pct
                        label = "Transcribing audio (CPU fallback)" if is_retry else "Transcribing audio"
                        progress_callback(
                            min(95.0, current_pct),
                            f"{label}: {segment.end:.1f}s / {duration:.1f}s ({min(95.0, current_pct):.1f}%)",
                        )
            return audio_segs, info

        try:
            audio_segments, info = _run_transcription(self._model)
        except RuntimeError as exc:
            logger.warning(
                "faster-whisper inference failed on current device: %s. Falling back to CPU.", exc
            )
            if progress_callback:
                progress_callback(25.0, "ASR device failed (e.g. missing CUDA library). Falling back to CPU...")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.num_threads,
            )
            audio_segments, info = _run_transcription(self._model, is_retry=True)

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
