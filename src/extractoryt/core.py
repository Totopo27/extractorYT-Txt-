"""Core YouTube transcript & audio extraction engine."""

import logging
import os
import re
import shutil
import tempfile
from typing import Any, Callable, Optional, Union
import httpx

from extractoryt.asr.base import AudioTranscriberPort
from extractoryt.asr.factory import get_transcriber
from extractoryt.exceptions import (
    ASRTranscriptionError,
    ExtractionError,
    InvalidYouTubeURLError,
    TranscriptNotFoundError,
)
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoMetadata

logger = logging.getLogger(__name__)

YOUTUBE_URL_PATTERN = re.compile(
    r"^(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|v\/|shorts\/|live\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)

YOUTUBE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")


class YouTubeExtractor:
    """
    High-fidelity extractor for YouTube transcripts and metadata with offline ASR fallback.
    """

    def __init__(
        self,
        http_timeout: float = 15.0,
        audio_transcriber: Optional[AudioTranscriberPort] = None,
        default_languages: Optional[list[str]] = None,
    ) -> None:
        self.http_timeout = http_timeout
        self.audio_transcriber = audio_transcriber
        self.default_languages = default_languages or ["es", "es-419", "en"]

    def is_youtube_url(self, url: str) -> bool:
        """Determines if a given URL or string matches a supported YouTube video pattern."""
        if not url:
            return False
        clean = url.strip()
        return bool(YOUTUBE_ID_PATTERN.match(clean) or YOUTUBE_URL_PATTERN.search(clean))

    def extract_video_id(self, url_or_id: str) -> Optional[str]:
        """Extracts the canonical 11-character video ID from a URL or raw ID."""
        if not url_or_id:
            return None
        clean = url_or_id.strip()
        if YOUTUBE_ID_PATTERN.match(clean):
            return clean
        match = YOUTUBE_URL_PATTERN.search(clean)
        return match.group(1) if match else None

    def fetch_metadata(self, video_id: str) -> VideoMetadata:
        """
        Retrieves video metadata (title, author, thumbnail) using the official public oEmbed API.
        No Google API key required.
        """
        target_url = f"https://www.youtube.com/watch?v={video_id}"
        oembed_url = f"https://www.youtube.com/oembed?url={target_url}&format=json"

        try:
            with httpx.Client(timeout=self.http_timeout, follow_redirects=True) as client:
                resp = client.get(oembed_url)
                if resp.status_code == 200:
                    data = resp.json()
                    return VideoMetadata(
                        video_id=video_id,
                        title=data.get("title") or f"YouTube Video ({video_id})",
                        author_name=data.get("author_name") or "YouTube Channel",
                        author_url=data.get("author_url") or "",
                        thumbnail_url=data.get("thumbnail_url")
                        or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                        video_url=target_url,
                    )
        except Exception as exc:
            logger.warning("Could not fetch oEmbed metadata for YouTube video %s: %s", video_id, exc)

        # Fallback metadata if oEmbed is unavailable
        return VideoMetadata(
            video_id=video_id,
            title=f"YouTube Video ({video_id})",
            author_name="YouTube",
            author_url="",
            thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            video_url=target_url,
        )

    def fetch_native_transcript(
        self,
        video_id: str,
        preferred_languages: Optional[list[str]] = None,
    ) -> tuple[list[TranscriptSnippet], str, bool]:
        """
        Fetches timestamped transcript snippets via youtube-transcript-api.
        Returns (snippets, language_code, is_generated).
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            import youtube_transcript_api._errors as yt_errors
        except ImportError as exc:
            raise ImportError(
                "youtube-transcript-api is required. Install with: pip install youtube-transcript-api"
            ) from exc

        langs = preferred_languages or self.default_languages

        try:
            # Instantiate or use static depending on installed library version
            if hasattr(YouTubeTranscriptApi, "list"):
                api = YouTubeTranscriptApi()
                transcript_list = api.list(video_id)
            elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]
            else:
                raise RuntimeError("Incompatible youtube-transcript-api interface.")

            chosen_transcript = None

            # 1. Try manually created transcript in preferred languages
            try:
                chosen_transcript = transcript_list.find_manually_created_transcript(langs)
            except Exception:
                pass

            # 2. Try auto-generated transcript in preferred languages
            if not chosen_transcript:
                try:
                    chosen_transcript = transcript_list.find_generated_transcript(langs)
                except Exception:
                    pass

            # 3. Try any available transcript in preferred languages
            if not chosen_transcript:
                try:
                    chosen_transcript = transcript_list.find_transcript(langs)
                except Exception:
                    pass

            # 4. Fallback to the very first available transcript
            if not chosen_transcript:
                for t in transcript_list:
                    chosen_transcript = t
                    break

            if not chosen_transcript:
                raise TranscriptNotFoundError(f"No transcripts found for YouTube video ({video_id}).")

            fetched = chosen_transcript.fetch()
            if hasattr(fetched, "to_raw_data"):
                raw_entries = fetched.to_raw_data()
            elif isinstance(fetched, list):
                raw_entries = fetched
            else:
                raw_entries = list(fetched)

            lang = getattr(chosen_transcript, "language_code", "unknown")
            is_gen = getattr(chosen_transcript, "is_generated", False)

            snippets: list[TranscriptSnippet] = []
            for item in raw_entries:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                start = float(item.get("start", 0.0))
                duration = float(item.get("duration", 0.0))
                snippets.append(
                    TranscriptSnippet(
                        start=start,
                        duration=duration,
                        text=text,
                    )
                )

            return snippets, lang, is_gen

        except (yt_errors.TranscriptsDisabled, yt_errors.NoTranscriptFound) as exc:
            logger.warning("Transcripts disabled or missing for YouTube video %s: %s", video_id, exc)
            raise TranscriptNotFoundError(
                f"YouTube video ({video_id}) has transcripts disabled or missing."
            ) from exc
        except Exception as exc:
            if isinstance(exc, TranscriptNotFoundError):
                raise
            logger.error("Error retrieving transcripts for %s: %s", video_id, exc)
            raise TranscriptNotFoundError(
                f"Failed to fetch transcript for video ({video_id}): {exc}"
            ) from exc

    def download_audio_stream(self, video_id: str) -> bytes:
        """
        Downloads the audio-only stream of a YouTube video using yt-dlp.
        Returns the raw audio bytes without requiring external ffmpeg installed.
        """
        try:
            import yt_dlp
        except ImportError as exc:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp") from exc

        temp_dir = tempfile.mkdtemp(prefix="extractoryt_audio_")
        target_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": target_template,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                downloaded_file = ydl.prepare_filename(info)
                with open(downloaded_file, "rb") as f_in:
                    return f_in.read()
        except Exception as exc:
            raise ExtractionError(
                f"Failed to download audio stream for video ({video_id}): {exc}"
            ) from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def extract(
        self,
        url_or_id: str,
        preferred_languages: Optional[list[str]] = None,
        fallback_to_asr: bool = True,
        force_asr: bool = False,
        asr_engine: str = "sherpa",
        asr_model: str = "small",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ExtractionResult:
        """
        Executes complete extraction workflow:
        1. Resolves video ID & fetches metadata.
        2. Tries native transcripts unless `force_asr` is True.
        3. If native fails and `fallback_to_asr` is True (or `force_asr` is True),
           downloads audio stream and runs offline ASR speech-to-text.
        """
        video_id = self.extract_video_id(url_or_id)
        if not video_id:
            raise InvalidYouTubeURLError(f"Invalid YouTube URL or video ID: '{url_or_id}'")

        if progress_callback:
            progress_callback(5.0, f"Fetching metadata for video {video_id}...")

        metadata = self.fetch_metadata(video_id)

        snippets: list[TranscriptSnippet] = []
        language = "en"
        is_generated = False
        source = "native"

        if not force_asr:
            try:
                if progress_callback:
                    progress_callback(10.0, "Checking native subtitles...")
                snippets, language, is_generated = self.fetch_native_transcript(
                    video_id=video_id,
                    preferred_languages=preferred_languages,
                )
                source = "native_generated" if is_generated else "native_manual"
                if progress_callback:
                    progress_callback(100.0, "Native transcript retrieved successfully.")
                return ExtractionResult(
                    metadata=metadata,
                    snippets=snippets,
                    language=language,
                    is_generated=is_generated,
                    source=source,
                )
            except TranscriptNotFoundError as err:
                if not fallback_to_asr:
                    raise err
                logger.info(
                    "Native transcripts unavailable for video %s. Initiating ASR fallback...",
                    video_id,
                )

        # Fallback or Forced ASR Flow
        transcriber = self.audio_transcriber
        if transcriber is None:
            transcriber = get_transcriber(engine=asr_engine, model_size=asr_model)

        if progress_callback:
            progress_callback(15.0, "Downloading audio stream with yt-dlp...")

        try:
            audio_bytes = self.download_audio_stream(video_id)

            if progress_callback:
                progress_callback(30.0, "Transcribing audio with ASR engine...")

            target_lang = preferred_languages[0] if preferred_languages else None
            asr_result = transcriber.transcribe(
                audio_bytes,
                progress_callback=progress_callback,
                language=target_lang,
            )

            snippets = [
                TranscriptSnippet(
                    start=seg.start_seconds,
                    duration=max(0.1, seg.end_seconds - seg.start_seconds),
                    text=seg.text,
                    speaker=seg.speaker,
                )
                for seg in asr_result.segments
            ]
            language = asr_result.language or "en"
            is_generated = True
            source = f"asr_{asr_engine.lower()}"

            return ExtractionResult(
                metadata=metadata,
                snippets=snippets,
                language=language,
                is_generated=is_generated,
                source=source,
            )
        except Exception as exc:
            if isinstance(exc, ExtractionError):
                raise
            raise ASRTranscriptionError(
                f"Offline ASR transcription failed for video ({video_id}): {exc}"
            ) from exc
