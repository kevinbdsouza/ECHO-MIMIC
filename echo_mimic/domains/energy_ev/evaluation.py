"""Evaluation helpers for EV charging heuristics and nudges."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .scenario import EVScenario, enumerate_global_optimum, enumerate_local_optima, compute_global_cost


def _run_python_script(script_path: Path, workdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", script_path.name],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )


def evaluate_local_policy_script(
    code: str,
    *,
    scenario: EVScenario,
    scenario_dir: Path,
    output_filename: str = "local_policy_output.json",
) -> Tuple[float, Dict[str, object]]:
    """Execute candidate code and score imitation accuracy."""

    script_path = scenario_dir / "_candidate_local.py"
    script_path.write_text(code, encoding="utf-8")

    result = _run_python_script(script_path, scenario_dir)
    if result.returncode != 0:
        return 0.0, {
            "status": "error",
            "stderr": result.stderr,
        }

    output_path = scenario_dir / output_filename
    if not output_path.exists():
        return 0.0, {"status": "missing_output"}

    with output_path.open("r", encoding="utf-8") as handle:
        allocation = json.load(handle)

    if not isinstance(allocation, Sequence) or len(allocation) != scenario.num_agents:
        return 0.0, {"status": "invalid_shape", "output": allocation}

    local_optima = enumerate_local_optima(scenario)
    correct = 0
    per_agent = {}
    for idx, choice in enumerate(allocation, start=1):
        try:
            choice_int = int(choice)
        except (ValueError, TypeError):
            return 0.0, {"status": "non_integer", "agent": idx, "value": choice}
        best_slots = local_optima[idx]
        per_agent[str(idx)] = {
            "choice": choice_int,
            "best": best_slots,
            "match": choice_int in best_slots,
        }
        if choice_int in best_slots:
            correct += 1

    accuracy = correct / scenario.num_agents
    return accuracy, {
        "status": "ok",
        "accuracy": accuracy,
        "per_agent": per_agent,
    }


def evaluate_global_policy_script(
    code: str,
    *,
    scenario: EVScenario,
    scenario_dir: Path,
    output_filename: str = "global_policy_output.json",
) -> Tuple[float, Dict[str, object]]:
    """Score a policy by negative global cost (higher is better)."""

    script_path = scenario_dir / "_candidate_global.py"
    script_path.write_text(code, encoding="utf-8")

    result = _run_python_script(script_path, scenario_dir)
    if result.returncode != 0:
        return float("-inf"), {
            "status": "error",
            "stderr": result.stderr,
        }

    output_path = scenario_dir / output_filename
    if not output_path.exists():
        return float("-inf"), {"status": "missing_output"}

    with output_path.open("r", encoding="utf-8") as handle:
        allocation = json.load(handle)

    if not isinstance(allocation, Sequence) or len(allocation) != scenario.num_agents:
        return float("-inf"), {"status": "invalid_shape", "output": allocation}

    try:
        allocation_int = [int(slot) for slot in allocation]
    except (ValueError, TypeError):
        return float("-inf"), {"status": "non_integer", "output": allocation}

    global_cost = compute_global_cost(scenario, allocation_int)
    best_allocation, best_score = enumerate_global_optimum(scenario)

    return -global_cost, {
        "status": "ok",
        "allocation": allocation_int,
        "global_cost": global_cost,
        "best_score": best_score,
        "regret": global_cost - best_score,
        "best_allocation": best_allocation,
    }


def evaluate_nudge_response(
    message: str,
    *,
    scenario: EVScenario,
    recommended_allocation: Sequence[int],
) -> Tuple[float, Dict[str, object]]:
    """Validate a JSON nudge response and check recommended slot alignment."""

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        return 0.0, {"status": "invalid_json", "error": str(exc)}

    required_keys = {"persona", "recommended_slot", "message"}
    if not required_keys.issubset(payload):
        return 0.0, {"status": "missing_keys", "payload": payload}

    try:
        recommended_slot = int(payload["recommended_slot"])
    except (ValueError, TypeError):
        return 0.0, {"status": "bad_slot", "value": payload["recommended_slot"]}

    persona = str(payload["persona"])
    text = str(payload["message"])

    agent_map = {agent.persona: idx for idx, agent in enumerate(scenario.agents)}
    if persona not in agent_map:
        return 0.0, {"status": "unknown_persona", "persona": persona}

    agent_index = agent_map[persona]
    target_slot = int(recommended_allocation[agent_index])

    score = 1.0 if recommended_slot == target_slot else 0.0
    detail = {
        "status": "ok" if score > 0 else "mismatch",
        "persona": persona,
        "agent_index": agent_index + 1,
        "recommended_slot": recommended_slot,
        "target_slot": target_slot,
        "message": text,
    }
    return score, detail

