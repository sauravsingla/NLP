"""Small, dependency-free text preprocessing utilities."""

from __future__ import annotations

import html
import re
import unicodedata

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    mask_emails: bool = True,
) -> str:
    """Normalize user-provided text without destroying punctuation or meaning.

    Args:
        text: Input string to normalize.
        lowercase: Convert the result to lowercase.
        remove_urls: Remove HTTP and ``www`` URLs.
        mask_emails: Replace email addresses with ``[email]``.

    Returns:
        A normalized string with Unicode and whitespace standardized.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", html.unescape(text))
    if remove_urls:
        normalized = _URL_PATTERN.sub(" ", normalized)
    if mask_emails:
        normalized = _EMAIL_PATTERN.sub(" [email] ", normalized)
    if lowercase:
        normalized = normalized.lower()

    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()
