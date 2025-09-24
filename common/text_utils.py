"""Utilities for parsing structured text returned by LLMs."""

from __future__ import annotations

import re
from typing import Optional

_PYTHON_CODE_BLOCK_RE = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)
_COMMUNICATION_BLOCK_RE = re.compile(r"\\communication\{([^}]+)\}")
_CODE_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(text: str, *, fallback_to_input: bool = True) -> str:
    """Return the contents of the first fenced python block in *text*.

    Args:
        text: Raw model response that may include fenced code blocks.
        fallback_to_input: When ``True`` and no python block exists, return the
            original ``text``; otherwise return an empty string.
    """
    if not text:
        return ""

    match = _PYTHON_CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    if not fallback_to_input:
        return ""
    return text.strip()


def extract_message(text: str) -> str:
    """Extract the payload of a ``\\communication{...}`` block."""
    if not text:
        return ""
    match = _COMMUNICATION_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def strip_code_fences(text: str) -> str:
    """Remove generic triple-backtick fenced blocks from *text*."""
    if not text:
        return ""
    return _CODE_FENCE_RE.sub(lambda m: m.group(1).strip(), text).strip()


__all__ = [
    "extract_python_code",
    "extract_message",
    "strip_code_fences",
]
