"""Core YouTube transcript & audio extraction engine."""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Union
import httpx

from extractoryt.asr.base import AudioTranscriberPort
from extractoryt.asr.factory import get_transcriber
from extractoryt.cleaner import clean_sound_artifacts
from extractoryt.exceptions import (
    ASRTranscriptionError,
    ExtractionError,
    InvalidYouTubeURLError,
    TranscriptNotFoundError,
)
from extractoryt.models import ExtractionResult, TranscriptSnippet, VideoChapter, VideoMetadata

logger = logging.getLogger(__name__)

YOUTUBE_URL_PATTERN = re.compile(
    r"^(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|v\/|shorts\/|live\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)

YOUTUBE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")

GDRIVE_URL_PATTERN = re.compile(
    r"^(?:https?:\/\/)?(?:drive\.google\.com\/(?:(?:u\/\d+\/)?file\/d\/|open\?(?:.*&)?id=|uc\?(?:.*&)?id=|drive\/folders\/)|docs\.google\.com\/(?:(?:u\/\d+\/)?file\/d\/))([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)

ONEDRIVE_URL_PATTERN = re.compile(
    r"^(?:https?:\/\/)?(?:1drv\.ms\/|onedrive\.live\.com\/|(?:[a-zA-Z0-9_-]+\.)*sharepoint\.com\/)",
    re.IGNORECASE,
)


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
        clean_noise: bool = True,
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
                if clean_noise:
                    text = clean_sound_artifacts(text)
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

    def fetch_chapters(
        self,
        video_id: str,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
    ) -> list[VideoChapter]:
        """
        Retrieves chapters for a YouTube video using yt-dlp metadata inspection.
        Returns empty list if yt-dlp is not available, fails, or no chapters exist.
        """
        try:
            import yt_dlp
        except ImportError:
            return []

        ydl_opts: dict[str, Any] = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser.lower(), None, None, None)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                if not info:
                    return []
                if "entries" in info:
                    entries = [e for e in info.get("entries", []) if e]
                    info = entries[0] if entries else {}
                raw_chapters = info.get("chapters") or []
                chapters: list[VideoChapter] = []
                for c in raw_chapters:
                    chapters.append(
                        VideoChapter(
                            title=str(c.get("title", "")),
                            start=float(c.get("start_time", 0.0)),
                            end=float(c.get("end_time", 0.0)),
                        )
                    )
                return chapters
        except Exception as exc:
            logger.debug("Failed to fetch chapters for video %s: %s", video_id, exc)
            return []

    def download_audio_stream(
        self,
        video_id: str,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> bytes:
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
        ydl_opts: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": target_template,
            "quiet": True,
            "no_warnings": True,
        }
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser.lower(), None, None, None)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                if metadata is not None and not metadata.chapters and info:
                    raw_chapters = info.get("chapters") or []
                    if raw_chapters:
                        metadata.chapters = [
                            VideoChapter(
                                title=str(c.get("title", "")),
                                start=float(c.get("start_time", 0.0)),
                                end=float(c.get("end_time", 0.0)),
                            )
                            for c in raw_chapters
                        ]
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
        device: str = "auto",
        clean_noise: bool = True,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        extract_chapters: bool = True,
        **kwargs: Any,
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
                    clean_noise=clean_noise,
                )
                source = "native_generated" if is_generated else "native_manual"

                if extract_chapters and not metadata.chapters:
                    if progress_callback:
                        progress_callback(85.0, "Fetching video chapters...")
                    metadata.chapters = self.fetch_chapters(
                        video_id=video_id,
                        cookie_file=cookie_file,
                        cookies_from_browser=cookies_from_browser,
                    )

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
            transcriber = get_transcriber(
                engine=asr_engine,
                model_size=asr_model,
                device=device,
            )

        if progress_callback:
            progress_callback(15.0, "Downloading audio stream with yt-dlp...")

        try:
            audio_bytes = self.download_audio_stream(
                video_id,
                cookie_file=cookie_file,
                cookies_from_browser=cookies_from_browser,
                metadata=metadata if extract_chapters else None,
            )

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
                    text=clean_sound_artifacts(seg.text) if clean_noise else seg.text,
                    speaker=seg.speaker,
                )
                for seg in asr_result.segments
                if not clean_noise or clean_sound_artifacts(seg.text)
            ]
            language = asr_result.language or "en"
            is_generated = True
            source = f"asr_{asr_engine.lower()}"

            if extract_chapters and not metadata.chapters:
                metadata.chapters = self.fetch_chapters(
                    video_id=video_id,
                    cookie_file=cookie_file,
                    cookies_from_browser=cookies_from_browser,
                )

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


class LocalMediaExtractor:
    """
    High-fidelity extractor for local media files (MP4, MKV, MP3, WAV, etc.)
    using PyAV for metadata inspection and offline ASR for transcription.
    """

    def __init__(
        self,
        audio_transcriber: Optional[AudioTranscriberPort] = None,
        default_languages: Optional[list[str]] = None,
    ) -> None:
        self.audio_transcriber = audio_transcriber
        self.default_languages = default_languages or ["es", "es-419", "en"]

    def read_metadata(self, file_path: Union[str, Path]) -> tuple[VideoMetadata, Optional[str]]:
        """
        Reads local media file metadata using PyAV.
        Returns a tuple of (VideoMetadata, container_format_name).
        """
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "PyAV (av) is required for local media extraction. "
                "Install it with: pip install av"
            ) from exc

        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local media file not found: '{file_path}'")

        filename = path.name
        file_stem = path.stem
        duration: Optional[float] = None
        format_name: Optional[str] = None

        try:
            with av.open(str(path)) as container:
                audio_stream = next((s for s in container.streams if s.type == "audio"), None)
                if not audio_stream:
                    raise ExtractionError(f"No audio stream found in local media file: '{path}'")

                if container.duration is not None and container.duration > 0:
                    duration = float(container.duration) / 1_000_000.0

                if container.format and container.format.name:
                    format_name = container.format.name
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Failed to inspect media file '{path}' with av: {exc}") from exc

        metadata = VideoMetadata(
            video_id=filename,
            title=file_stem,
            author_name="Local Media",
            video_url=str(file_path),
            duration_seconds=duration,
        )
        return metadata, format_name

    def extract(
        self,
        file_path: Union[str, Path],
        preferred_languages: Optional[list[str]] = None,
        asr_engine: str = "sherpa",
        asr_model: str = "small",
        device: str = "auto",
        clean_noise: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Extracts transcript from a local media file using the specified ASR engine.
        """
        str_path = str(file_path)
        if not os.path.isfile(str_path):
            raise FileNotFoundError(f"Local media file not found: '{file_path}'")

        path = Path(file_path)
        filename = path.name
        file_stem = path.stem

        if progress_callback:
            progress_callback(5.0, f"Inspecting local media file '{filename}'...")

        metadata, _ = self.read_metadata(file_path)

        transcriber = self.audio_transcriber
        if transcriber is None:
            transcriber = get_transcriber(
                engine=asr_engine,
                model_size=asr_model,
                device=device,
            )

        if progress_callback:
            progress_callback(15.0, f"Transcribing audio with {asr_engine} engine...")

        langs = preferred_languages or self.default_languages
        target_lang = langs[0] if langs else None

        try:
            asr_result = transcriber.transcribe(
                str_path,
                progress_callback=progress_callback,
                language=target_lang,
            )

            snippets = [
                TranscriptSnippet(
                    start=seg.start_seconds,
                    duration=max(0.1, seg.end_seconds - seg.start_seconds),
                    text=clean_sound_artifacts(seg.text) if clean_noise else seg.text,
                    speaker=seg.speaker,
                )
                for seg in asr_result.segments
                if not clean_noise or clean_sound_artifacts(seg.text)
            ]

            if metadata.duration_seconds is None or metadata.duration_seconds <= 0:
                if asr_result.duration_seconds > 0:
                    metadata.duration_seconds = asr_result.duration_seconds
                elif snippets:
                    metadata.duration_seconds = snippets[-1].end

            language = asr_result.language or (target_lang or "en")
            is_generated = True
            source = f"local_{asr_engine.lower()}"

            if progress_callback:
                progress_callback(100.0, "Local media transcription completed successfully.")

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
                f"Offline ASR transcription failed for local media file ({filename}): {exc}"
            ) from exc


class CloudMediaExtractor:
    """
    High-fidelity extractor for cloud media storage (Google Drive, OneDrive / SharePoint)
    using yt-dlp for audio streaming and offline ASR for transcription.
    """

    def __init__(
        self,
        audio_transcriber: Optional[AudioTranscriberPort] = None,
        default_languages: Optional[list[str]] = None,
    ) -> None:
        self.audio_transcriber = audio_transcriber
        self.default_languages = default_languages or ["es", "es-419", "en"]

    @staticmethod
    def is_cloud_url(url: str) -> bool:
        """Determines if a given URL or string matches a supported cloud storage pattern."""
        if not url:
            return False
        clean = url.strip()
        return bool(GDRIVE_URL_PATTERN.search(clean) or ONEDRIVE_URL_PATTERN.search(clean))

    @staticmethod
    def detect_service(url: str) -> str:
        """Identifies the cloud storage service ("gdrive", "onedrive", or "cloud")."""
        if not url:
            return "cloud"
        clean = url.strip()
        if GDRIVE_URL_PATTERN.search(clean):
            return "gdrive"
        if ONEDRIVE_URL_PATTERN.search(clean):
            return "onedrive"
        return "cloud"

    def download_audio_stream(
        self,
        url: str,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
    ) -> tuple[bytes, VideoMetadata]:
        """
        Downloads the audio-only stream from a cloud URL using yt-dlp.
        Returns the raw audio bytes and extracted VideoMetadata.
        """
        try:
            import yt_dlp
        except ImportError as exc:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp") from exc

        service = self.detect_service(url)
        temp_dir = tempfile.mkdtemp(prefix="extractoryt_cloud_")
        target_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
        ydl_opts: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": target_template,
            "quiet": True,
            "no_warnings": True,
        }
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
        if cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser.lower(), None, None, None)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise ExtractionError(f"No media information could be extracted from cloud URL: '{url}'")

                if "entries" in info:
                    entries = [e for e in info.get("entries", []) if e]
                    if not entries:
                        raise ExtractionError(f"No media entries found at cloud URL: '{url}'")
                    entry_info = entries[0]
                else:
                    entry_info = info

                downloaded_file = ydl.prepare_filename(entry_info)
                if not os.path.isfile(downloaded_file):
                    files = [
                        os.path.join(temp_dir, f)
                        for f in os.listdir(temp_dir)
                        if os.path.isfile(os.path.join(temp_dir, f))
                    ]
                    if files:
                        downloaded_file = files[0]
                    else:
                        raise ExtractionError(
                            f"Downloaded cloud audio file not found in temporary directory: {temp_dir}"
                        )

                with open(downloaded_file, "rb") as f_in:
                    audio_bytes = f_in.read()

                media_id = str(entry_info.get("id") or "cloud_media")
                default_title = (
                    "Google Drive Media" if service == "gdrive" else (
                        "OneDrive Media" if service == "onedrive" else "Cloud Media"
                    )
                )
                title = entry_info.get("title") or default_title
                default_author = (
                    "Google Drive" if service == "gdrive" else (
                        "OneDrive" if service == "onedrive" else "Cloud Media"
                    )
                )
                author_name = entry_info.get("uploader") or entry_info.get("channel") or default_author
                duration = (
                    float(entry_info["duration"])
                    if entry_info.get("duration") is not None
                    else None
                )
                thumbnail = entry_info.get("thumbnail") or ""
                webpage_url = entry_info.get("webpage_url") or url

                raw_chapters = entry_info.get("chapters") or []
                chapters: list[VideoChapter] = [
                    VideoChapter(
                        title=str(c.get("title", "")),
                        start=float(c.get("start_time", 0.0)),
                        end=float(c.get("end_time", 0.0)),
                    )
                    for c in raw_chapters
                ]

                metadata = VideoMetadata(
                    video_id=media_id,
                    title=title,
                    author_name=author_name,
                    author_url=entry_info.get("uploader_url") or "",
                    thumbnail_url=thumbnail,
                    video_url=webpage_url,
                    duration_seconds=duration,
                    chapters=chapters,
                )
                return audio_bytes, metadata
        except Exception as exc:
            if isinstance(exc, ExtractionError):
                raise
            raise ExtractionError(
                f"Failed to download cloud audio stream from ({url}): {exc}"
            ) from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def extract(
        self,
        url: str,
        preferred_languages: Optional[list[str]] = None,
        asr_engine: str = "sherpa",
        asr_model: str = "small",
        device: str = "auto",
        clean_noise: bool = True,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Downloads cloud audio stream and transcribes via ASR engine.
        """
        if not self.is_cloud_url(url):
            raise ValueError(f"Invalid cloud media URL: '{url}'")

        service = self.detect_service(url)
        service_display = (
            "Google Drive" if service == "gdrive" else ("OneDrive" if service == "onedrive" else "Cloud")
        )

        if progress_callback:
            progress_callback(5.0, f"Downloading audio from {service_display} with yt-dlp...")

        audio_bytes, metadata = self.download_audio_stream(
            url,
            cookie_file=cookie_file,
            cookies_from_browser=cookies_from_browser,
        )

        transcriber = self.audio_transcriber
        if transcriber is None:
            transcriber = get_transcriber(
                engine=asr_engine,
                model_size=asr_model,
                device=device,
            )

        if progress_callback:
            progress_callback(30.0, f"Transcribing audio with {asr_engine} engine...")

        langs = preferred_languages or self.default_languages
        target_lang = langs[0] if langs else None

        try:
            asr_result = transcriber.transcribe(
                audio_bytes,
                progress_callback=progress_callback,
                language=target_lang,
            )

            snippets = [
                TranscriptSnippet(
                    start=seg.start_seconds,
                    duration=max(0.1, seg.end_seconds - seg.start_seconds),
                    text=clean_sound_artifacts(seg.text) if clean_noise else seg.text,
                    speaker=seg.speaker,
                )
                for seg in asr_result.segments
                if not clean_noise or clean_sound_artifacts(seg.text)
            ]

            if metadata.duration_seconds is None or metadata.duration_seconds <= 0:
                if asr_result.duration_seconds > 0:
                    metadata.duration_seconds = asr_result.duration_seconds
                elif snippets:
                    metadata.duration_seconds = snippets[-1].end

            language = asr_result.language or (target_lang or "en")
            is_generated = True
            source = f"{service}_{asr_engine.lower()}"

            if progress_callback:
                progress_callback(100.0, f"{service_display} media transcription completed successfully.")

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
                f"Offline ASR transcription failed for cloud media ({url}): {exc}"
            ) from exc

    extract_cloud_media = extract


class MediaExtractor:
    """
    Unified dispatcher / facade for extracting transcripts from local media files,
    YouTube, or cloud storage (Google Drive, OneDrive / SharePoint).
    """

    def __init__(
        self,
        youtube_extractor: Optional[YouTubeExtractor] = None,
        local_extractor: Optional[LocalMediaExtractor] = None,
        cloud_extractor: Optional[CloudMediaExtractor] = None,
        audio_transcriber: Optional[AudioTranscriberPort] = None,
        default_languages: Optional[list[str]] = None,
    ) -> None:
        self.default_languages = default_languages or ["es", "es-419", "en"]
        self.youtube_extractor = youtube_extractor or YouTubeExtractor(
            audio_transcriber=audio_transcriber,
            default_languages=self.default_languages,
        )
        self.local_extractor = local_extractor or LocalMediaExtractor(
            audio_transcriber=audio_transcriber,
            default_languages=self.default_languages,
        )
        self.cloud_extractor = cloud_extractor or CloudMediaExtractor(
            audio_transcriber=audio_transcriber,
            default_languages=self.default_languages,
        )

    def is_local_file(self, source: str) -> bool:
        """Determines if the provided source points to an existing local file."""
        return bool(source and os.path.isfile(source.strip()))

    def is_youtube_url(self, source: str) -> bool:
        """Determines if the provided source matches a YouTube URL or video ID."""
        return bool(source and self.youtube_extractor.is_youtube_url(source.strip()))

    def is_cloud_url(self, source: str) -> bool:
        """Determines if the provided source matches a Google Drive or OneDrive URL."""
        return bool(source and self.cloud_extractor.is_cloud_url(source.strip()))

    def extract(
        self,
        source: str,
        preferred_languages: Optional[list[str]] = None,
        fallback_to_asr: bool = True,
        force_asr: bool = False,
        asr_engine: str = "sherpa",
        asr_model: str = "small",
        device: str = "auto",
        clean_noise: bool = True,
        cookie_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        extract_chapters: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Inspects the source string.
        If is_local_file -> delegates to LocalMediaExtractor.
        If is_youtube_url -> delegates to YouTubeExtractor.
        If is_cloud_url -> delegates to CloudMediaExtractor.
        Else raises ValueError or appropriate error.
        """
        if not source:
            raise ValueError("Media source cannot be empty.")

        clean_source = source.strip()

        if self.is_local_file(clean_source):
            return self.local_extractor.extract(
                file_path=clean_source,
                preferred_languages=preferred_languages,
                asr_engine=asr_engine,
                asr_model=asr_model,
                device=device,
                clean_noise=clean_noise,
                progress_callback=progress_callback,
                extract_chapters=extract_chapters,
                **kwargs,
            )
        elif self.is_youtube_url(clean_source):
            return self.youtube_extractor.extract(
                url_or_id=clean_source,
                preferred_languages=preferred_languages,
                fallback_to_asr=fallback_to_asr,
                force_asr=force_asr,
                asr_engine=asr_engine,
                asr_model=asr_model,
                device=device,
                clean_noise=clean_noise,
                cookie_file=cookie_file,
                cookies_from_browser=cookies_from_browser,
                progress_callback=progress_callback,
                extract_chapters=extract_chapters,
                **kwargs,
            )
        elif self.is_cloud_url(clean_source):
            return self.cloud_extractor.extract(
                url=clean_source,
                preferred_languages=preferred_languages,
                asr_engine=asr_engine,
                asr_model=asr_model,
                device=device,
                clean_noise=clean_noise,
                cookie_file=cookie_file,
                cookies_from_browser=cookies_from_browser,
                progress_callback=progress_callback,
                extract_chapters=extract_chapters,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported media source: '{source}'. Source must be an existing local file, a valid YouTube URL/ID, or a supported cloud URL (Google Drive, OneDrive)."
            )


YouTubeTranscriptExtractor = YouTubeExtractor

