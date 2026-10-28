"""Command-line interface (CLI) for extractorYT-Txt."""

import argparse
import os
import sys
from typing import Optional

from extractoryt.core import MediaExtractor, YouTubeExtractor
from extractoryt.exceptions import ExtractionError
from extractoryt.formatters import (
    JsonFormatter,
    MarkdownFormatter,
    PlainTextFormatter,
    SrtFormatter,
    VttFormatter,
)
from extractoryt.models import ExtractionResult

__version__ = "0.1.0"


def _format_result(
    result: ExtractionResult,
    format_type: str,
    include_timestamps: bool = False,
) -> str:
    """Renders the extraction result into the requested format string."""
    fmt = format_type.lower()
    if fmt in ("md", "markdown"):
        return MarkdownFormatter().format(result)
    elif fmt in ("txt", "text", "plain"):
        return PlainTextFormatter(include_timestamps=include_timestamps).format(result)
    elif fmt == "json":
        return JsonFormatter().format(result)
    elif fmt == "srt":
        return SrtFormatter().format(result)
    elif fmt == "vtt":
        return VttFormatter().format(result)
    else:
        raise ValueError(f"Unsupported format: '{format_type}'")


def create_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for extractorYT CLI."""
    parser = argparse.ArgumentParser(
        prog="extractoryt",
        description="High-fidelity YouTube, Google Drive, OneDrive & audio-to-text extraction with offline ASR fallback.",
    )
    parser.add_argument(
        "source",
        help="YouTube video URL/ID, cloud storage URL (Google Drive, OneDrive), or local media file path (MP4, MKV, MP3, WAV, etc.).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["markdown", "text", "json", "srt", "vtt"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file path (default: '-' for stdout).",
    )
    parser.add_argument(
        "-l",
        "--lang",
        default="es,en",
        help="Comma-separated preferred language codes (default: 'es,en').",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable offline ASR fallback if native subtitles are unavailable.",
    )
    parser.add_argument(
        "--force-asr",
        action="store_true",
        help="Force offline ASR transcription regardless of native subtitle availability.",
    )
    parser.add_argument(
        "--asr-engine",
        choices=["sherpa", "faster-whisper"],
        default="sherpa",
        help="ASR engine for offline fallback (default: sherpa).",
    )
    parser.add_argument(
        "--asr-model",
        default="small",
        help="Model size for ASR engine (e.g. tiny, base, small; default: small).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device for ASR engine (choices: auto, cuda, cpu; default: auto).",
    )
    parser.add_argument(
        "--cookies",
        help="Path to cookies.txt file for authenticated extraction (Google Drive, OneDrive, YouTube).",
    )
    parser.add_argument(
        "--cookies-from-browser",
        help="Extract cookies from a browser (e.g. chrome, firefox, edge, brave) for authenticated access.",
    )
    parser.add_argument(
        "--keep-sound-tags",
        action="store_true",
        help="Retain artificial YouTube sound markers like [Música] or [Aplausos].",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamp anchors in plain text output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress and status messages, outputting only the transcript.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"extractoryt {__version__}",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint function."""
    parser = create_parser()
    args = parser.parse_args(argv)

    languages = [lang.strip() for lang in args.lang.split(",") if lang.strip()]

    # Setup rich console if available and not quiet
    use_rich = False
    console = None
    if not args.quiet:
        try:
            from rich.console import Console

            console = Console(stderr=True)
            use_rich = True
        except ImportError:
            pass

    def log_status(message: str) -> None:
        if args.quiet:
            return
        if use_rich and console:
            console.print(f"[bold cyan]•[/bold cyan] {message}")
        else:
            sys.stderr.write(f"• {message}\n")
            sys.stderr.flush()

    def progress_callback(pct: float, message: str) -> None:
        if args.quiet:
            return
        if use_rich and console:
            console.print(f"  [dim][{pct:5.1f}%][/dim] {message}")
        else:
            sys.stderr.write(f"  [{pct:5.1f}%] {message}\n")
            sys.stderr.flush()

    is_local = os.path.isfile(args.source)
    extractor = MediaExtractor(default_languages=languages)

    if is_local:
        log_status(f"Processing local media file target: '{args.source}'")
    elif extractor.is_cloud_url(args.source):
        log_status(f"Processing cloud media target: '{args.source}'")
    else:
        log_status(f"Processing YouTube video target: '{args.source}'")

    try:
        result = extractor.extract(
            source=args.source,
            preferred_languages=languages,
            fallback_to_asr=not args.no_fallback,
            force_asr=args.force_asr,
            asr_engine=args.asr_engine,
            asr_model=args.asr_model,
            device=args.device,
            clean_noise=not args.keep_sound_tags,
            cookie_file=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
            progress_callback=progress_callback if not args.quiet else None,
        )

        formatted_output = _format_result(
            result=result,
            format_type=args.format,
            include_timestamps=args.timestamps,
        )

        if args.output == "-":
            sys.stdout.write(formatted_output)
            sys.stdout.flush()
        else:
            with open(args.output, "w", encoding="utf-8") as f_out:
                f_out.write(formatted_output)

            log_status(f"Saved transcript to: '{args.output}'")
            if use_rich and console:
                from rich.table import Table

                table = Table(title="Extraction Summary", show_header=True, header_style="bold magenta")
                table.add_column("Property", style="dim")
                table.add_column("Value")
                table.add_row("Title", result.metadata.title)
                table.add_row("Author", result.metadata.author_name)
                table.add_row("Language", result.language)
                table.add_row("Source", result.source)
                table.add_row("Snippets", str(len(result.snippets)))
                table.add_row("Output Path", args.output)
                console.print(table)

        return 0

    except (ExtractionError, ValueError, FileNotFoundError) as exc:
        if use_rich and console:
            console.print(f"[bold red]Extraction Error:[/bold red] {exc}")
        else:
            sys.stderr.write(f"Extraction Error: {exc}\n")
        return 1
    except Exception as exc:
        if use_rich and console:
            console.print(f"[bold red]Unexpected Error:[/bold red] {exc}")
        else:
            sys.stderr.write(f"Unexpected Error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
