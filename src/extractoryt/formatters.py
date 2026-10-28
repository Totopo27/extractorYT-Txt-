"""Formatters for structuring and exporting transcripts to multiple formats."""

import json
from typing import Optional

from extractoryt.models import ExtractionResult, TranscriptSnippet


def format_timestamp(
    seconds: float,
    srt_mode: bool = False,
    vtt_mode: bool = False,
) -> str:
    """
    Formats a duration in seconds into human-readable or subtitle timestamp formats.

    - Standard: MM:SS or HH:MM:SS
    - SRT mode: HH:MM:SS,mmm
    - VTT mode: HH:MM:SS.mmm
    """
    total_secs = max(0.0, float(seconds))
    hours = int(total_secs // 3600)
    minutes = int((total_secs % 3600) // 60)
    secs = int(total_secs % 60)
    millis = int(round((total_secs - int(total_secs)) * 1000.0))

    if millis >= 1000:
        millis = 999

    if srt_mode:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    if vtt_mode:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class MarkdownFormatter:
    """Renders transcripts as clean, structured, chunk-friendly Markdown documents."""

    def __init__(
        self,
        heading_interval_seconds: float = 300.0,
        paragraph_interval_seconds: float = 35.0,
        paragraph_char_limit: int = 350,
    ) -> None:
        self.heading_interval_seconds = heading_interval_seconds
        self.paragraph_interval_seconds = paragraph_interval_seconds
        self.paragraph_char_limit = paragraph_char_limit

    def format(self, result: ExtractionResult) -> str:
        """Formats the extraction result into structured Markdown."""
        meta = result.metadata
        duration_label = format_timestamp(result.duration_seconds)

        if result.source.startswith("local"):
            status_label = "(Transcribed from Local Media)"
        elif result.source.startswith("asr"):
            status_label = "(Transcribed via Offline ASR)"
        elif result.is_generated:
            status_label = "(Auto-generated Subtitles)"
        else:
            status_label = "(Verified Subtitles)"

        channel_line = (
            f"- **Channel / Speaker**: [{meta.author_name}]({meta.author_url})"
            if meta.author_url
            else f"- **Channel / Speaker**: {meta.author_name}"
        )

        source_url_line = (
            f"- **Source File**: `{meta.video_url}`"
            if result.source.startswith("local")
            else f"- **Original Video**: [{meta.video_url}]({meta.video_url})"
        )

        lines: list[str] = [
            f"# {meta.title}",
            "",
            channel_line,
            source_url_line,
            f"- **Language**: `{result.language}` {status_label}",
            f"- **Duration**: `{duration_label}`",
            f"- **Source**: `{result.source}`",
        ]

        if meta.thumbnail_url:
            lines.extend(["", f"![Video Thumbnail]({meta.thumbnail_url})"])

        lines.extend(["", "---", "", "## Structured Transcript", ""])

        if not result.snippets:
            lines.append("*(No transcript snippets found)*\n")
            return "\n".join(lines)

        current_paragraph_texts: list[str] = []
        current_paragraph_start = float(result.snippets[0].start)
        last_heading_time = -self.heading_interval_seconds

        for snippet in result.snippets:
            start = float(snippet.start)
            text = snippet.text.strip()
            if not text:
                continue

            # Heading break every heading_interval_seconds (default 5 minutes)
            if start - last_heading_time >= self.heading_interval_seconds:
                if current_paragraph_texts:
                    ts_str = format_timestamp(current_paragraph_start)
                    p_body = " ".join(current_paragraph_texts)
                    lines.append(f"**[{ts_str}]** {p_body}\n")
                    current_paragraph_texts = []

                header_ts = format_timestamp(start)
                lines.append(f"\n### [{header_ts}] Timestamp {header_ts}\n")
                last_heading_time = start
                current_paragraph_start = start

            current_paragraph_texts.append(text)
            accumulated_len = sum(len(t) for t in current_paragraph_texts)

            # Flush paragraph when interval exceeded or char limit reached
            if (
                (start - current_paragraph_start >= self.paragraph_interval_seconds)
                or (accumulated_len >= self.paragraph_char_limit)
            ):
                ts_str = format_timestamp(current_paragraph_start)
                p_body = " ".join(current_paragraph_texts)
                lines.append(f"**[{ts_str}]** {p_body}\n")
                current_paragraph_texts = []
                current_paragraph_start = start + float(snippet.duration)

        # Flush final trailing paragraph
        if current_paragraph_texts:
            ts_str = format_timestamp(current_paragraph_start)
            p_body = " ".join(current_paragraph_texts)
            lines.append(f"**[{ts_str}]** {p_body}\n")

        return "\n".join(lines).strip() + "\n"


class PlainTextFormatter:
    """Formats transcripts as plain text with optional timestamp prefixes."""

    def __init__(self, include_timestamps: bool = False) -> None:
        self.include_timestamps = include_timestamps

    def format(self, result: ExtractionResult) -> str:
        """Formats the extraction result into plain text."""
        if not self.include_timestamps:
            return result.full_text + "\n"

        lines: list[str] = []
        for snippet in result.snippets:
            text = snippet.text.strip()
            if text:
                ts = format_timestamp(snippet.start)
                lines.append(f"[{ts}] {text}")

        return "\n".join(lines) + ("\n" if lines else "")


class SrtFormatter:
    """Formats transcripts into standard SubRip (.srt) subtitle format."""

    def format(self, result: ExtractionResult) -> str:
        """Renders snippets into SubRip format."""
        blocks: list[str] = []
        for idx, snippet in enumerate(result.snippets, start=1):
            text = snippet.text.strip()
            if not text:
                continue
            start_srt = format_timestamp(snippet.start, srt_mode=True)
            end_srt = format_timestamp(snippet.end, srt_mode=True)
            blocks.append(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n")

        return "\n".join(blocks)


class VttFormatter:
    """Formats transcripts into standard WebVTT (.vtt) format."""

    def format(self, result: ExtractionResult) -> str:
        """Renders snippets into WebVTT format."""
        lines: list[str] = ["WEBVTT", ""]
        for idx, snippet in enumerate(result.snippets, start=1):
            text = snippet.text.strip()
            if not text:
                continue
            start_vtt = format_timestamp(snippet.start, vtt_mode=True)
            end_vtt = format_timestamp(snippet.end, vtt_mode=True)
            lines.append(f"{idx}")
            lines.append(f"{start_vtt} --> {end_vtt}")
            lines.append(f"{text}")
            lines.append("")

        return "\n".join(lines)


class JsonFormatter:
    """Formats transcripts into readable JSON representation."""

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def format(self, result: ExtractionResult) -> str:
        """Renders extraction result into indented JSON string."""
        return json.dumps(result.to_dict(), indent=self.indent, ensure_ascii=False) + "\n"
