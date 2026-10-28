# extractorYT-Txt

High-fidelity YouTube transcript & audio-to-text extraction pipeline with offline ASR fallback.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](pyproject.toml)
[![Architecture: Clean](https://img.shields.io/badge/Architecture-Clean-brightgreen.svg)](#architecture)

Decoupled and elevated from `openfolioLM`, `extractorYT-Txt` provides a production-grade extraction engine and CLI that prioritizes native YouTube transcripts and seamlessly falls back to local Speech-To-Text (ASR) via `yt-dlp` and `sherpa-onnx` (Whisper ONNX + Silero VAD) or `faster-whisper`.

---

## Key Features

- **Multi-Tier Extraction Strategy**:
  - **Priority 1 (Native Subtitles)**: Ultra-fast extraction of manual and auto-generated subtitles via `youtube-transcript-api` with prioritized language selection (e.g. `es`, `es-419`, `en`).
  - **Priority 2 (Offline ASR Fallback)**: Automatically downloads the audio stream via `yt-dlp` and runs offline speech recognition using `sherpa-onnx` (Silero VAD + Whisper ONNX int8) or `faster-whisper` (CTranslate2).
- **Clean Architecture**: Domain entities, ports, and formatters are fully decoupled from storage, CLI, and concrete ASR backends.
- **Multiple Output Formats**:
  - **Markdown (`.md`)**: Structured documents with video metadata headers, thumbnail links, periodic section headings (e.g., every 5 minutes), and readable paragraph grouping with timestamp anchors `**[MM:SS]**`.
  - **Plain Text (`.txt`)**: Raw transcript text or timestamp-prefixed lines.
  - **SubRip (`.srt`)**: Industry-standard subtitle format with sequential indexing and millisecond precision.
  - **WebVTT (`.vtt`)**: Web-compatible video subtitles format.
  - **JSON (`.json`)**: Machine-readable data structure containing full metadata, snippet timings, and speaker tags.
- **Modular Packaging**: Core transcript fetching is lightweight and fast. Heavy neural network dependencies (`onnxruntime`, `sherpa-onnx`, `ctranslate2`, `av`) are optional extras (`[asr]`, `[faster-whisper]`).
- **Rich CLI**: Interactive command-line utility with progress indicators, status tables, and stdout/file redirection.

---

## Installation

### 1. Lightweight Installation (Native Subtitles Only)
Only installs `youtube-transcript-api`, `httpx`, and `yt-dlp`:
```bash
pip install .
```

### 2. With Offline ASR Fallback (Sherpa-ONNX + Silero VAD)
Installs `sherpa-onnx`, `av` (PyAV), and `numpy`:
```bash
pip install ".[asr]"
```

### 3. With Faster-Whisper ASR
Installs `faster-whisper` (CTranslate2) and audio decoding tools:
```bash
pip install ".[faster-whisper]"
```

### 4. Full Installation (All Backends + Rich CLI + Testing)
```bash
pip install ".[all,dev]"
```

---

## Command-Line Usage (CLI)

The package installs the `extractoryt` command:

```bash
# Extract transcript as Markdown to stdout
extractoryt https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Save structured Markdown to a file
extractoryt https://www.youtube.com/watch?v=dQw4w9WgXcQ -o transcript.md

# Process a Google Drive video directly (streams audio only, no full video download)
extractoryt "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view?usp=sharing" -o drive_transcript.md

# Process a OneDrive / SharePoint video directly
extractoryt "https://1drv.ms/v/s!Ah3_example" -o onedrive_transcript.md

# Use browser cookies for private Drive / OneDrive / YouTube media
extractoryt "https://drive.google.com/file/d/PRIVATE_FILE_ID/view" --cookies-from-browser chrome -o private.md

# Export as SubRip subtitles (.srt)
extractoryt https://www.youtube.com/watch?v=dQw4w9WgXcQ -f srt -o subtitles.srt

# Export as structured JSON
extractoryt https://www.youtube.com/watch?v=dQw4w9WgXcQ -f json -o output.json

# Export as Plain Text with timestamps
extractoryt https://www.youtube.com/watch?v=dQw4w9WgXcQ -f text --timestamps

# Specify preferred languages (comma-separated, in order of priority)
extractoryt dQw4w9WgXcQ -l "es,es-419,en"

# Strict mode: Disable offline ASR fallback (fails if video has no subtitles)
extractoryt https://youtu.be/dQw4w9WgXcQ --no-fallback

# Force offline ASR transcription (bypasses native subtitles)
extractoryt https://youtu.be/dQw4w9WgXcQ --force-asr --asr-engine sherpa --asr-model small

# Quiet mode: suppress progress output (ideal for unix pipes)
extractoryt https://youtu.be/dQw4w9WgXcQ -q > transcript.md
```

### CLI Arguments Reference

| Option | Description | Default |
| ------ | ----------- | ------- |
| `source` | YouTube URL/ID, local media file, or Google Drive / OneDrive link | *(Required)* |
| `-f, --format` | Output format: `markdown`, `text`, `json`, `srt`, `vtt` | `markdown` |
| `-o, --output` | Target file path (`-` for stdout) | `-` |
| `-l, --lang` | Comma-separated preferred language codes | `es,en` |
| `--no-fallback` | Disable ASR fallback when native subtitles are absent | `False` |
| `--force-asr` | Force ASR transcription, ignoring native subtitles | `False` |
| `--asr-engine` | Engine for speech recognition: `sherpa` or `faster-whisper` | `sherpa` |
| `--asr-model` | Whisper model size (`tiny`, `base`, `small`, `medium`) | `small` |
| `--cookies` | Path to cookies.txt for private media authentication | `None` |
| `--cookies-from-browser` | Extract cookies from browser (`chrome`, `edge`, `firefox`, etc.) | `None` |
| `--timestamps` | Include timestamps in plain text format | `False` |
| `-q, --quiet` | Suppress status and progress output | `False` |
| `--version` | Display version and exit | |

---

## Python API Usage

You can embed `extractorYT-Txt` directly in any Python application or pipeline:

```python
from extractoryt import YouTubeExtractor, MarkdownFormatter, JsonFormatter

extractor = YouTubeExtractor(default_languages=["es", "en"])

# 1. High-fidelity extraction
result = extractor.extract(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    fallback_to_asr=True,  # Will download audio & transcribe if no subtitles
    asr_engine="sherpa",
    asr_model="small",
)

# 2. Access metadata and structured results
print(f"Title: {result.metadata.title}")
print(f"Author: {result.metadata.author_name}")
print(f"Duration: {result.duration_seconds}s")
print(f"Extraction Source: {result.source}")
print(f"Snippet Count: {len(result.snippets)}")

# 3. Format into desired output
markdown_doc = MarkdownFormatter().format(result)
json_data = JsonFormatter().format(result)
```

### Dependency Injection with Custom Transcribers

You can inject any custom audio transcriber conforming to the `AudioTranscriberPort` protocol:

```python
from extractoryt import YouTubeExtractor
from extractoryt.asr.base import AudioTranscriberPort, AudioTranscriptResult, AudioSegment

class CustomWhisperTranscriber(AudioTranscriberPort):
    def transcribe(self, audio_input, progress_callback=None, language=None):
        # Your custom model execution logic
        return AudioTranscriptResult(
            segments=[AudioSegment(start_seconds=0.0, end_seconds=5.0, text="Custom ASR text")],
            language="en",
            duration_seconds=5.0,
            full_text="Custom ASR text"
        )

extractor = YouTubeExtractor(audio_transcriber=CustomWhisperTranscriber())
result = extractor.extract("dQw4w9WgXcQ", force_asr=True)
```

---

## Project Structure

```
extractorYT-Txt/
├── pyproject.toml              # Modern build & packaging configuration
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── src/
│   └── extractoryt/
│       ├── __init__.py         # Package exports & version
│       ├── core.py             # YouTubeExtractor orchestration engine
│       ├── models.py           # Domain models: Snippet, Metadata, Result
│       ├── ports.py            # Interfaces: TranscriberPort, MetadataProviderPort
│       ├── formatters.py       # Formatters: Markdown, PlainText, JSON, SRT, VTT
│       ├── exceptions.py       # Domain-specific exceptions
│       ├── cli.py              # Rich CLI implementation
│       └── asr/
│           ├── __init__.py     # ASR module exports
│           ├── base.py         # AudioTranscriberPort & data contracts
│           ├── decoder.py      # PyAV audio to PCM float32 converter
│           ├── sherpa.py       # SherpaOnnxTranscriber with Silero VAD
│           ├── faster_whisper.py # FasterWhisperTranscriber
│           └── factory.py      # ASR engine factory
└── tests/
    ├── __init__.py
    ├── test_url_parser.py      # URL parsing & video ID extraction tests
    ├── test_formatters.py      # Output format & timestamp tests
    ├── test_extractor.py       # Orchestrator & native subtitles tests
    ├── test_asr.py             # ASR fallback & decoder tests
    └── test_cli.py             # CLI flags & I/O tests
```

---

## Testing

Run the automated test suite using `pytest`:

```bash
python -m pytest tests -v
```

All 29 tests execute unit and integration verification for URL parsing, formatters, oEmbed metadata handling, native subtitle prioritization, ASR fallbacks, audio decoding, and CLI parsing.

---

## License

This project is licensed under the MIT License.
