"""Shared constants for little_questions.

Single source of truth for sentence type labels and supported language codes.
"""

SENTENCE_TYPES: list[str] = ["command", "exclamation", "question", "request", "statement"]

SUPPORTED_LANGUAGES: list[str] = ["en", "es", "pt", "ca", "fr", "de", "it", "nl"]
