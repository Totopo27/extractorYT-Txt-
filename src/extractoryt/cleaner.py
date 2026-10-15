"""Text cleaning utilities to strip artificial YouTube sound markers while strictly preserving spoken words."""

import re

# Exact YouTube audio/sound tags injected automatically or by subtitlers
SOUND_TAG_PATTERN = re.compile(
    r"\[(?:Música|Musica|Music|Aplausos|Applause|Risas|Laughter|Silencio|Silence|Gritos|Screaming)\]",
    re.IGNORECASE,
)

# Collapses multiple consecutive spaces to a single space
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")


def clean_sound_artifacts(text: str) -> str:
    """
    Strips artificial sound tags (e.g. '[Música]', '[Aplausos]', '[Music]')
    from transcript snippets without altering or rephrasing spoken words.

    Returns:
        Cleaned text with extra spaces collapsed, or empty string if text consisted only of sound tags.
    """
    if not text:
        return ""

    # Remove known bracketed sound markers
    cleaned = SOUND_TAG_PATTERN.sub("", text)
    # Collapse multiple spaces leftover from removed tags
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned
