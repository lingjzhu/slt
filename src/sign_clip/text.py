from __future__ import annotations

import re
import unicodedata


_SEPARATOR_RE = re.compile(r"[_\-/]+")
_ALNUM_BOUNDARY_RE = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_LATIN_RE = re.compile(r"[a-z]")


def normalize_sign_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "").strip())
    text = _SEPARATOR_RE.sub(" ", text)
    text = _ALNUM_BOUNDARY_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    tokens = text.split()
    if len(tokens) > 1 and _LATIN_RE.search(text):
        while len(tokens) > 1 and tokens[-1].isdigit():
            tokens.pop()
    return " ".join(tokens)


def extract_text_label(metadata: dict) -> str:
    for key in ("transcription", "gloss", "caption", "text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_sign_text(value)
    return ""
