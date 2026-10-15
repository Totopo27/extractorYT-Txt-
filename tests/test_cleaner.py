"""Unit tests for the clean_sound_artifacts utility."""

import pytest
from extractoryt.cleaner import clean_sound_artifacts


def test_clean_sound_artifacts_empty():
    assert clean_sound_artifacts("") == ""
    assert clean_sound_artifacts(None) == ""


def test_clean_sound_artifacts_only_tags():
    assert clean_sound_artifacts("[Música]") == ""
    assert clean_sound_artifacts("[Música] [Música]") == ""
    assert clean_sound_artifacts("[Music]   [Aplausos]") == ""
    assert clean_sound_artifacts("[Risas]") == ""


def test_clean_sound_artifacts_preserves_speech():
    raw = "he [Música] Buenas tardes Antes que todo"
    cleaned = clean_sound_artifacts(raw)
    assert cleaned == "he Buenas tardes Antes que todo"


def test_clean_sound_artifacts_mixed_tags_and_whitespace():
    raw = "   Hola a todos [Aplausos] bienvenidos a esta charla [Música]   "
    cleaned = clean_sound_artifacts(raw)
    assert cleaned == "Hola a todos bienvenidos a esta charla"


def test_clean_sound_artifacts_does_not_modify_normal_words():
    raw = "Esta es una frase completamente normal sin corchetes de sonido."
    assert clean_sound_artifacts(raw) == raw
