"""Shared helpers for configuring Gemini/OpenAI models and rate limiting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

from ..config import Config
from .rate_limiter import RateLimiter

_DEFAULT_ENV_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
_API_KEY: Optional[str] = None


def configure_genai(api_key_env_keys: Iterable[str] = _DEFAULT_ENV_KEYS) -> str:
    """Load environment variables and configure the Gemini SDK once."""
    global _API_KEY
    if _API_KEY:
        return _API_KEY

    load_dotenv()
    for candidate in api_key_env_keys:
        api_key = os.getenv(candidate)
        if api_key:
            genai.configure(api_key=api_key)
            _API_KEY = api_key
            return api_key

    raise ValueError(
        "No Gemini API key found. Set one of: {}".format(
            ", ".join(api_key_env_keys)
        )
    )


def is_openai_model(model_name: Optional[str]) -> bool:
    """Return True if the requested model should be served via OpenAI."""
    if not model_name:
        return False
    lowered = model_name.lower()
    return lowered.startswith("gpt")


class _TextPart:
    """Minimal response part wrapper to mirror Gemini's structure."""

    def __init__(self, text: str):
        self.text = text


class _OpenAIResponse:
    """Adapter that exposes ``.text`` and ``.parts`` like Gemini responses."""

    def __init__(self, text: str):
        self.text = text
        self.parts = [_TextPart(text)]


def _normalize_history(history: Optional[Sequence[object]]) -> List[Dict[str, object]]:
    messages: List[Dict[str, object]] = []
    if not history:
        return messages
    for entry in history:
        normalized = _normalize_history_entry(entry)
        if normalized:
            messages.append(normalized)
    return messages


def _content_type_for_role(role: str) -> str:
    """Return the Responses API content type based on speaker role."""
    if role.lower() in ("assistant",):
        return "output_text"
    return "input_text"


def _normalize_history_entry(entry: object) -> Optional[Dict[str, object]]:
    if isinstance(entry, dict):
        role = str(entry.get("role", "user"))
        texts: List[str] = []
        if "parts" in entry:
            texts.extend(_extract_texts(entry["parts"]))
        if "content" in entry:
            texts.extend(_extract_texts(entry["content"]))
        if "text" in entry and entry["text"]:
            texts.append(str(entry["text"]))
        if not texts and "message" in entry:
            texts.append(str(entry["message"]))
        if texts:
            content_type = _content_type_for_role(role)
            return {
                "role": role,
                "content": [
                    {"type": content_type, "text": text}
                    for text in texts
                ],
            }
    elif isinstance(entry, str):
        return {
            "role": "user",
            "content": [
                {"type": _content_type_for_role("user"), "text": entry}
            ],
        }
    return None


def _extract_texts(parts: object) -> List[str]:
    texts: List[str] = []
    if isinstance(parts, str):
        texts.append(parts)
    elif isinstance(parts, Sequence):
        for part in parts:
            if isinstance(part, dict):
                text_value = part.get("text")
                if text_value:
                    texts.append(str(text_value))
            else:
                text_value = getattr(part, "text", None)
                if text_value:
                    texts.append(str(text_value))
    return texts


class OpenAIChatSession:
    """Simple chat session wrapper that mimics the Gemini client interface."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: str,
        history: Optional[Sequence[object]] = None,
    ):
        self._client = client
        self._model_name = model_name
        self._history: List[Dict[str, Any]] = []
        if system_instruction:
            self._history.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": _content_type_for_role("system"),
                            "text": system_instruction,
                        }
                    ],
                }
            )
        self._history.extend(_normalize_history(history))

    def send_message(self, message: str) -> _OpenAIResponse:
        self._history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": _content_type_for_role("user"),
                        "text": message,
                    }
                ],
            }
        )
        response = self._client.responses.create(
            model=self._model_name,
            input=self._history,
        )
        text = _collect_openai_text(response)
        self._history.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": _content_type_for_role("assistant"),
                        "text": text,
                    }
                ],
            }
        )
        return _OpenAIResponse(text)


class OpenAIChatModel:
    """Adapter exposing ``start_chat`` compatible with Gemini's API."""

    def __init__(self, model_name: str, system_instruction: str):
        if OpenAI is None:  # pragma: no cover - import guard
            raise RuntimeError("openai package is not installed")
        self._client = OpenAI()
        self._model_name = model_name
        self._system_instruction = system_instruction

    def start_chat(self, *, history: Optional[Sequence[object]] = None):
        return OpenAIChatSession(
            self._client,
            self._model_name,
            self._system_instruction,
            history=history,
        )


def _collect_openai_text(response: Any) -> str:
    texts: List[str] = []
    outputs = getattr(response, "output", None) or []
    for output in outputs:
        if not output:
            continue
        contents = getattr(output, "content", None) or []
        for content in contents:
            text = getattr(content, "text", None)
            if text:
                texts.append(text)
    fallback = getattr(response, "output_text", None)
    if fallback:
        texts.append(fallback)
    if not texts:
        texts.append("")
    return "\n".join(texts).strip()


def build_model(model_name: str, system_instruction: str, *, ensure_configured: bool = True):
    """Create a chat model compatible with the ``GenerativeModel`` API."""
    if is_openai_model(model_name):
        return OpenAIChatModel(model_name, system_instruction)
    if ensure_configured:
        configure_genai()
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )


def ensure_rate_limiter(cfg: Optional[Config] = None) -> RateLimiter:
    """Instantiate a rate limiter using configuration defaults."""
    configuration = cfg or Config()
    return RateLimiter(**configuration.rate_limit)


@dataclass(frozen=True)
class GeminiModels:
    """Container for the trio of policy/farm/fix models some workflows use."""

    policy: Optional[genai.GenerativeModel] = None
    farm: Optional[genai.GenerativeModel] = None
    fix: Optional[genai.GenerativeModel] = None


def init_models(
    *,
    policy: Optional[Tuple[str, str]] = None,
    farm: Optional[Tuple[str, str]] = None,
    fix: Optional[Tuple[str, str]] = None,
) -> GeminiModels:
    """Convenience helper to instantiate commonly paired models.

    Each tuple should be ``(model_name, system_instruction)``.
    Missing tuples leave the corresponding attribute as ``None``.
    """
    if any(
        pair and not is_openai_model(pair[0])
        for pair in (policy, farm, fix)
    ):
        configure_genai()

    def _build(pair: Optional[Tuple[str, str]]):
        if not pair:
            return None
        name, instructions = pair
        return build_model(name, instructions, ensure_configured=False)

    return GeminiModels(
        policy=_build(policy),
        farm=_build(farm),
        fix=_build(fix),
    )


__all__ = [
    "GeminiModels",
    "build_model",
    "configure_genai",
    "ensure_rate_limiter",
    "init_models",
    "is_openai_model",
]
