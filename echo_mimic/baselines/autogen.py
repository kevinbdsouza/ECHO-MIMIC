"""AutoGen-inspired multi-agent baseline."""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from echo_mimic.models import build_model_client

logger = logging.getLogger(__name__)


@dataclass
class AutoGenRequest:
    mode: str
    agent_id: Optional[str]
    model: Optional[str]
    data_hint: Optional[str]


@dataclass
class AutoGenConversation:
    planner_draft: str
    critique: str
    refined_plan: str
    log_path: str


class AutoGenBaseline:
    """Coordinate a minimal AutoGen-like planner/critic loop."""

    def __init__(self, domain: str, log_dir: str = "outputs/autogen") -> None:
        self.domain = domain
        self.log_dir = log_dir

    def run(
        self, mode: str, agent_id: Optional[str], model: Optional[str], data_hint: str
    ) -> None:
        request = AutoGenRequest(mode=mode, agent_id=agent_id, model=model, data_hint=data_hint)
        logger.info(
            "[AutoGen] Running domain=%s mode=%s agent=%s model=%s",
            self.domain,
            request.mode,
            request.agent_id,
            request.model,
        )
        conversation = self._coordinate_agents(request)
        logger.info("[AutoGen] Plan ready. Transcript stored at %s", conversation.log_path)

    def _coordinate_agents(self, request: AutoGenRequest) -> AutoGenConversation:
        model_name = request.model or os.getenv("GENAI_MODEL", "gemini-flash-lite-latest")
        client = build_model_client(model_name)
        data_summary = self._summarize_data_hint(request.data_hint)

        planner_prompt = (
            f"You are PlannerAgent ({request.agent_id or 'autogen-planner'}) working inside a multi-agent AutoGen loop. "
            f"Domain: {self.domain}. Mode: {request.mode}.\n"
            f"Data summary: {data_summary}.\n"
            "Draft a numbered plan with 3-6 concrete actions, each including the file you would inspect and the expected output."
        )
        planner_draft = self._generate_with_fallback(client, planner_prompt)

        critic_prompt = (
            "You are CriticAgent reviewing a teammate's plan for weaknesses. "
            "List potential risks, missing validations, and any clarifying questions."
            f"\nPlan to review:\n{planner_draft}"
        )
        critique = self._generate_with_fallback(client, critic_prompt)

        refinement_prompt = (
            f"You are PlannerAgent ({request.agent_id or 'autogen-planner'}) updating the plan after critique. "
            "Incorporate the feedback explicitly and return a final plan plus next-step checklist."
            f"\nOriginal plan:\n{planner_draft}\n\nCritique:\n{critique}"
        )
        refined_plan = self._generate_with_fallback(client, refinement_prompt)

        log_path = self._write_transcript(
            request=request,
            planner_draft=planner_draft,
            critique=critique,
            refined_plan=refined_plan,
        )

        logger.info("[AutoGen][Planner] Draft plan:\n%s", planner_draft)
        logger.info("[AutoGen][Critic] Feedback:\n%s", critique)
        logger.info("[AutoGen][Planner] Refined plan:\n%s", refined_plan)

        return AutoGenConversation(
            planner_draft=planner_draft,
            critique=critique,
            refined_plan=refined_plan,
            log_path=log_path,
        )

    def _generate_with_fallback(self, client, prompt: str) -> str:  # pragma: no cover - depends on model availability
        try:
            return client.generate(prompt)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("AutoGen generation failed (%s). Falling back to static text.", exc)
            return (
                "1) Inspect available data files.\n"
                "2) Draft a simple heuristic and peer review it.\n"
                "3) Validate outputs and record open questions."
            )

    def _summarize_data_hint(self, data_hint: Optional[str]) -> str:
        if not data_hint:
            return "no data hint provided"
        matches = sorted(glob.glob(data_hint))
        if not matches:
            return f"no files matched hint '{data_hint}'"
        folders = [path for path in matches if os.path.isdir(path)]
        if folders:
            sample = ", ".join(os.path.basename(f) for f in folders[:5])
            return f"{len(folders)} folders ({sample}...)"
        sample_files = ", ".join(os.path.basename(f) for f in matches[:5])
        return f"{len(matches)} files ({sample_files}...)"

    def _write_transcript(
        self, *, request: AutoGenRequest, planner_draft: str, critique: str, refined_plan: str
    ) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        filename = f"{self.domain}_{request.mode}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.md"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# AutoGen session for {self.domain} ({request.mode})\n\n")
            fh.write(f"Agent ID: {request.agent_id or 'autogen-agent'}\n")
            fh.write(f"Model: {request.model or os.getenv('GENAI_MODEL', 'gemini-flash-lite-latest')}\n")
            fh.write(f"Data hint: {request.data_hint}\n\n")
            fh.write("## Planner draft\n")
            fh.write(planner_draft.strip() + "\n\n")
            fh.write("## Critic feedback\n")
            fh.write(critique.strip() + "\n\n")
            fh.write("## Refined plan\n")
            fh.write(refined_plan.strip() + "\n")
        return path


__all__ = ["AutoGenBaseline"]
