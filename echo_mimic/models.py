"""Model factory supporting Gemini and OpenAI chat models."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except Exception:  # pragma: no cover - handled at runtime
    genai = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class ModelClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class GeminiClient:
    model_name: str

    def generate(self, prompt: str) -> str:
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        completion = genai.GenerativeModel(self.model_name).generate_content(prompt)
        return completion.text or ""


@dataclass
class OpenAIClient:
    model_name: str

    def generate(self, prompt: str) -> str:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        client = OpenAI()
        response = client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": prompt}],
        )
        choice = response.output[0].content[0].text
        return getattr(choice, "value", "") or getattr(choice, "text", "")


def build_model_client(model_name: str) -> ModelClient:
    if model_name.startswith("gpt") or model_name.startswith("o1"):
        logger.info("Using OpenAI client for model %s", model_name)
        return OpenAIClient(model_name)
    logger.info("Using Gemini client for model %s", model_name)
    return GeminiClient(model_name)
