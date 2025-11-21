"""Helpers for repairing model-generated Python code."""

from __future__ import annotations

import re
from typing import Callable

from .rate_limiter import RateLimiter, send_message_with_retry

from .text_utils import extract_python_code, strip_code_fences


_MODEL_ARTIFACT_RE = re.compile(r"<ctrl\d+>")


def _clean_fixed_code(raw_text: str, extractor: Callable[[str], str]) -> str:
    code = extractor(raw_text) or raw_text
    code = strip_code_fences(code)
    code = _MODEL_ARTIFACT_RE.sub("", code)
    return code.strip()


def fix_with_model(
    fix_model,
    code: str,
    trace: str,
    *,
    rate_limiter: RateLimiter,
    code_extractor: Callable[[str], str] = extract_python_code,
    include_code_fence_hint: bool = False,
) -> str:
    """Ask the ``fix_model`` to repair ``code`` given ``trace``."""
    session = fix_model.start_chat(history=[])

    fence_hint = " Give the full code." if not include_code_fence_hint else (
        " Give the full code in a ```python ... ``` block."
    )

    prompt = (
        "Given the following Python code:\n\n"\
        "```python\n" + code + "\n```\n"
        "And the following traceback:\n" + trace + "\n"
        "Fix the errors and return the correct functioning python code." + fence_hint + "\n"
    )

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    return _clean_fixed_code(response, code_extractor)


__all__ = ["fix_with_model"]
