"""Offline Speech-To-Text Transcriber using sherpa-onnx (Whisper ONNX + Silero VAD)."""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union
import httpx
import numpy as np

from extractoryt.asr.base import AudioSegment, AudioTranscriptResult, AudioTranscriberPort
from extractoryt.asr.decoder import decode_audio_to_pcm

logger = logging.getLogger(__name__)

HF_WHISPER_BASE_URL = "https://huggingface.co/csukuangfj"
GITHUB_VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

MODEL_FILENAMES = {
    "tiny": {
        "repo": "sherpa-onnx-whisper-tiny",
        "encoder": "tiny-encoder.int8.onnx",
        "decoder": "tiny-decoder.int8.onnx",
        "tokens": "tiny-tokens.txt",
    },
    "base": {
        "repo": "sherpa-onnx-whisper-base",
        "encoder": "base-encoder.int8.onnx",
        "decoder": "base-decoder.int8.onnx",
        "tokens": "base-tokens.txt",
    },
    "small": {
        "repo": "sherpa-onnx-whisper-small",
        "encoder": "small-encoder.int8.onnx",
        "decoder": "small-decoder.int8.onnx",
        "tokens": "small-tokens.txt",
    },
}


class SherpaOnnxTranscriber(AudioTranscriberPort):
    """
    High-performance offline audio transcription using sherpa-onnx.
    Combines Silero VAD (Voice Activity Detection) with Whisper ONNX int8 models.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        model_size: str = "small",
        num_threads: int = 4,
        default_language: str = "",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if models_dir:
            self.models_dir = str(models_dir)
        else:
            cache_home = os.environ.get(
                "EXTRACTORYT_CACHE_DIR",
                os.path.join(str(Path.home()), ".cache", "extractoryt", "models", "sherpa"),
            )
            self.models_dir = cache_home

        self.model_size = model_size.lower()
        self.num_threads = num_threads
        self.default_language = default_language
        self.device = device
        self._recognizer: Optional[Any] = None
        self._vad: Optional[Any] = None

    def _download_file(
        self,
        url: str,
        destination_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Downloads a model file with bounded streaming and creates directories if needed."""
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        filename = os.path.basename(destination_path)
        logger.info("Downloading model component '%s' from %s...", filename, url)

        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", 0))
                downloaded_bytes = 0
                temp_dest = destination_path + ".tmp"

                with open(temp_dest, "wb") as f_out:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f_out.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total_bytes > 0 and progress_callback:
                            pct = (downloaded_bytes / total_bytes) * 100.0
                            progress_callback(pct, f"Downloading {filename} ({pct:.1f}%)...")

                if os.path.exists(destination_path):
                    os.remove(destination_path)
                os.rename(temp_dest, destination_path)
        logger.info("Component '%s' downloaded successfully.", filename)

    def _ensure_models(
        self, progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[str, str, str, str]:
        """Ensures that VAD and Whisper ONNX models exist locally; downloads if missing."""
        os.makedirs(self.models_dir, exist_ok=True)
        vad_path = os.path.join(self.models_dir, "silero_vad.onnx")

        if not os.path.exists(vad_path):
            if progress_callback:
                progress_callback(5.0, "Downloading Silero VAD model...")
            self._download_file(GITHUB_VAD_URL, vad_path, progress_callback)

        model_meta = MODEL_FILENAMES.get(self.model_size, MODEL_FILENAMES["small"])
        repo_name = model_meta["repo"]
        encoder_path = os.path.join(self.models_dir, model_meta["encoder"])
        decoder_path = os.path.join(self.models_dir, model_meta["decoder"])
        tokens_path = os.path.join(self.models_dir, model_meta["tokens"])

        base_hf_url = f"{HF_WHISPER_BASE_URL}/{repo_name}/resolve/main"

        if not os.path.exists(tokens_path):
            self._download_file(f"{base_hf_url}/{model_meta['tokens']}", tokens_path, progress_callback)
        if not os.path.exists(encoder_path):
            self._download_file(f"{base_hf_url}/{model_meta['encoder']}", encoder_path, progress_callback)
        if not os.path.exists(decoder_path):
            self._download_file(f"{base_hf_url}/{model_meta['decoder']}", decoder_path, progress_callback)

        return vad_path, encoder_path, decoder_path, tokens_path

    def _init_engines(
        self, progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """Lazily initializes the VAD and Whisper recognizer."""
        if self._recognizer is not None and self._vad is not None:
            return

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is not installed. "
                "Install it with: pip install extractoryt[asr] or pip install sherpa-onnx"
            ) from exc

        vad_path, encoder_path, decoder_path, tokens_path = self._ensure_models(progress_callback)

        # Initialize Silero VAD
        vad_config = sherpa_onnx.SileroVadModelConfig(
            model=vad_path,
            threshold=0.5,
            min_silence_duration=0.5,
            min_speech_duration=0.25,
            window_size=512,
            max_speech_duration=25.0,
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config=sherpa_onnx.VadModelConfig(silero_vad=vad_config, sample_rate=16000),
            buffer_size_in_seconds=60.0,
        )

        # Initialize Whisper Recognizer
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            language=self.default_language,
            task="transcribe",
            num_threads=self.num_threads,
        )

    def transcribe(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        language: Optional[str] = None,
    ) -> AudioTranscriptResult:
        """
        Transcribes an audio file, raw bytes, or raw float32 PCM numpy array.
        Uses VAD to segment into speech intervals and transcribes each segment.
        """
        self._init_engines(progress_callback)

        if isinstance(audio_input, (str, bytes)):
            if progress_callback:
                progress_callback(10.0, "Decoding audio stream...")
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
            progress_callback(20.0, "Segmenting speech with Silero VAD...")

        self._vad.reset()
        window_size = 512
        raw_segments: list[tuple[float, float, np.ndarray]] = []

        for i in range(0, len(samples), window_size):
            chunk = samples[i : i + window_size]
            if len(chunk) < window_size:
                padded = np.zeros(window_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded

            self._vad.accept_waveform(chunk)
            while not self._vad.empty():
                seg = self._vad.front
                start_sec = float(seg.start) / 16000.0
                end_sec = start_sec + (float(len(seg.samples)) / 16000.0)
                raw_segments.append((start_sec, end_sec, np.array(seg.samples, dtype=np.float32)))
                self._vad.pop()

        self._vad.flush()
        while not self._vad.empty():
            seg = self._vad.front
            start_sec = float(seg.start) / 16000.0
            end_sec = start_sec + (float(len(seg.samples)) / 16000.0)
            raw_segments.append((start_sec, end_sec, np.array(seg.samples, dtype=np.float32)))
            self._vad.pop()

        # Fallback windowing if VAD found no segments but duration > 0.5s
        if not raw_segments and duration > 0.5:
            step = 16000 * 20
            for i in range(0, len(samples), step):
                chunk = samples[i : i + step]
                start_sec = float(i) / 16000.0
                end_sec = float(i + len(chunk)) / 16000.0
                raw_segments.append((start_sec, end_sec, chunk))

        audio_segments: list[AudioSegment] = []
        total_segs = len(raw_segments)
        detected_language = language or self.default_language or "en"

        for idx, (start_sec, end_sec, seg_samples) in enumerate(raw_segments):
            if progress_callback and total_segs > 0:
                pct = 25.0 + (float(idx) / float(total_segs)) * 70.0
                progress_callback(pct, f"Transcribing segment {idx + 1} of {total_segs}...")

            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, seg_samples)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if hasattr(stream.result, "lang") and stream.result.lang:
                detected_language = stream.result.lang

            if text:
                audio_segments.append(
                    AudioSegment(
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        text=text,
                    )
                )

        if progress_callback:
            progress_callback(100.0, "Transcription completed.")

        full_text = " ".join(s.text for s in audio_segments)
        return AudioTranscriptResult(
            segments=audio_segments,
            language=detected_language,
            duration_seconds=duration,
            full_text=full_text,
        )
