"""Helpers for instruction-writing and evaluation agent creation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from echo_mimic.models import build_model_client


@dataclass
class PromptAuthoringRequest:
    task_description: str
    input_schema: str
    objectives: List[str]
    evaluation_criteria: List[str]
    model: str = "gemini-flash-lite-latest"
    agent_id: Optional[str] = None


def generate_prompts(request: PromptAuthoringRequest) -> Dict[str, str]:
    """Autogenerate system/agent/evaluation prompts from a structured request."""
    client = build_model_client(request.model)
    base = (
        "You are an instruction-writing agent."
        f" Task: {request.task_description}."
        f" Inputs: {request.input_schema}."
        f" Objectives: {', '.join(request.objectives)}."
        f" Evaluation criteria: {', '.join(request.evaluation_criteria)}."
    )
    system_prompt = client.generate(base + "\nReturn a crisp system prompt for downstream agents.")
    agent_prompt = client.generate(base + "\nOutline agent operating instructions in bullet form.")
    evaluation_prompt = client.generate(base + "\nDescribe how an evaluator should score outputs.")
    return {
        "system": system_prompt,
        "agent": agent_prompt,
        "evaluation": evaluation_prompt,
    }


@dataclass
class EvaluationHarnessRequest:
    objectives: List[str]
    metrics: List[str]
    acceptance_criteria: List[str]
    model: str = "gemini-flash-lite-latest"


def build_evaluation_harness(request: EvaluationHarnessRequest) -> Dict[str, str]:
    client = build_model_client(request.model)
    instructions = (
        "You are an evaluation harness designer. "
        f"Objectives: {json.dumps(request.objectives)}. "
        f"Metrics: {json.dumps(request.metrics)}. "
        f"Acceptance criteria: {json.dumps(request.acceptance_criteria)}."
    )
    evaluator = client.generate(instructions + "\nDesign a JSON schema for scoring outputs.")
    hooks = client.generate(instructions + "\nDescribe hooks to integrate into any task runner.")
    return {"schema": evaluator, "hooks": hooks}
