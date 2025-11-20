"""Lightweight AutoGen-style baseline.

This is intentionally minimal so that it can run without additional setup while
still providing an agentic baseline distinct from DSPy and the native
ECHO-MIMIC strategies.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from echo_mimic.models import build_model_client

logger = logging.getLogger(__name__)


@dataclass
class AutoGenRequest:
    mode: str
    agent_id: Optional[str]
    model: Optional[str]
    data_hint: Optional[str]


class AutoGenBaseline:
    def __init__(self, domain: str) -> None:
        self.domain = domain

    def run(
        self, mode: str, agent_id: Optional[str], model: Optional[str], data_hint: str
    ) -> None:
        request = AutoGenRequest(mode=mode, agent_id=agent_id, model=model, data_hint=data_hint)
        logger.info("[AutoGen] Running domain=%s mode=%s agent=%s model=%s", self.domain, request.mode, request.agent_id, request.model)
        self._generate_stub_response(request)

    def _generate_stub_response(self, request: AutoGenRequest) -> None:
        """Generate a simple plan using the configured model.

        The output is logged to keep the baseline non-intrusive while still
        demonstrating how OpenAI/Gemini models can be plugged into the pipeline.
        """
        prompt = (
            f"You are AutoGen agent {request.agent_id or 'autogen-agent'} preparing a plan for "
            f"domain={self.domain}, mode={request.mode}. Data hint: {request.data_hint}. "
            "List the next three high level actions you would take."
        )
        model_name = request.model or os.getenv("GENAI_MODEL", "gemini-flash-lite-latest")
        client = build_model_client(model_name)
        try:
            text = client.generate(prompt)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("AutoGen generation failed (%s). Falling back to static plan.", exc)
            text = "1) Inspect data. 2) Build heuristic. 3) Evaluate results."
        logger.info("[AutoGen] Proposed actions:\n%s", text)


__all__ = ["AutoGenBaseline"]
