"""Shared helpers for configuring Gemini models and rate limiting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from ..config import Config
from ..rate_limiter import RateLimiter

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


def build_model(model_name: str, system_instruction: str, *, ensure_configured: bool = True):
    """Create a ``GenerativeModel`` after ensuring the API key is loaded."""
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
]
