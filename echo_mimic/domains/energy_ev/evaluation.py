"""Evaluation helpers for EV charging heuristics and nudges."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .scenario import (
    EVScenario,
    enumerate_global_optimum,
    enumerate_local_optima,
    compute_global_cost,
)


def _run_python_script(script_path: Path, workdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", script_path.name],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )


def _coerce_schedule(payload: Sequence[object], *, num_days: int) -> List[int]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("Schedule must be a sequence of day-level slots")
    if len(payload) != num_days:
        raise ValueError("Schedule length must match number of days")
    try:
        return [int(value) for value in payload]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-integer slot detected: {payload}") from exc


def _transpose_agent_schedules(
    schedules: Sequence[Sequence[int]], *, num_days: int
) -> List[List[int]]:
    return [
        [schedule[day_idx] for schedule in schedules]
        for day_idx in range(num_days)
    ]


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
    total = scenario.num_agents * scenario.num_days
    per_agent: Dict[str, object] = {}
    for idx, schedule in enumerate(allocation, start=1):
        try:
            schedule_int = _coerce_schedule(schedule, num_days=scenario.num_days)
        except ValueError as exc:
            return 0.0, {"status": "invalid_schedule", "agent": idx, "error": str(exc)}
        best_slots = local_optima[idx]
        day_details = []
        for day_idx, slot in enumerate(schedule_int):
            best_for_day = best_slots[day_idx]
            match = slot in best_for_day
            if match:
                correct += 1
            day_details.append(
                {
                    "day": day_idx + 1,
                    "slot": slot,
                    "best": best_for_day,
                    "match": match,
                }
            )
        per_agent[str(idx)] = day_details

    accuracy = correct / total if total else 0.0
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

    schedules: List[List[int]] = []
    for idx, schedule in enumerate(allocation, start=1):
        try:
            schedule_int = _coerce_schedule(schedule, num_days=scenario.num_days)
        except ValueError as exc:
            return float("-inf"), {"status": "invalid_schedule", "agent": idx, "error": str(exc)}
        schedules.append(schedule_int)

    daily_allocation = _transpose_agent_schedules(schedules, num_days=scenario.num_days)
    global_cost = compute_global_cost(scenario, daily_allocation)
    best_allocation, best_score = enumerate_global_optimum(scenario)

    return -global_cost, {
        "status": "ok",
        "allocation": schedules,
        "global_cost": global_cost,
        "best_score": best_score,
        "regret": global_cost - best_score,
        "best_allocation": best_allocation,
    }


def evaluate_nudge_response(
    message: str,
    *,
    scenario: EVScenario,
    recommended_allocation: Sequence[Sequence[int]],
) -> Tuple[float, Dict[str, object]]:
    """Validate a JSON nudge response and check recommended slot alignment."""

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        return 0.0, {"status": "invalid_json", "error": str(exc)}

    required_keys = {"persona", "recommended_slots", "message"}
    if not required_keys.issubset(payload):
        return 0.0, {"status": "missing_keys", "payload": payload}

    try:
        recommended_slots = _coerce_schedule(payload["recommended_slots"], num_days=scenario.num_days)
    except (ValueError, TypeError, KeyError) as exc:
        return 0.0, {"status": "bad_slots", "error": str(exc)}

    persona = str(payload["persona"])
    text = str(payload["message"])

    agent_map = {agent.persona: idx for idx, agent in enumerate(scenario.agents)}
    if persona not in agent_map:
        return 0.0, {"status": "unknown_persona", "persona": persona}

    agent_index = agent_map[persona]
    target_schedule = [int(day[agent_index]) for day in recommended_allocation]
    matches = [slot == target for slot, target in zip(recommended_slots, target_schedule)]
    score = sum(matches) / scenario.num_days if scenario.num_days else 0.0
    detail = {
        "status": "ok" if score >= 1.0 else "partial",
        "persona": persona,
        "agent_index": agent_index + 1,
        "recommended_slots": recommended_slots,
        "target_slots": target_schedule,
        "message": text,
    }
    return score, detail


def evaluate_local_agent_policy_script(
    code: str,
    *,
    scenario: EVScenario,
    scenario_dir: Path,
    agent_id: int,
    output_filename: str = "local_policy_output.json",
) -> Tuple[float, Dict[str, object]]:
    """Execute and score an imitation policy for a single agent."""

    agent_dir = scenario_dir / "local" / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    script_path = agent_dir / "_candidate_local.py"
    script_path.write_text(code, encoding="utf-8")

    result = _run_python_script(script_path, agent_dir)
    if result.returncode != 0:
        return 0.0, {"status": "error", "stderr": result.stderr}

    output_path = agent_dir / output_filename
    if not output_path.exists():
        return 0.0, {"status": "missing_output"}

    with output_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            return 0.0, {"status": "invalid_json", "error": str(exc)}

    try:
        schedule = _coerce_schedule(payload, num_days=scenario.num_days)
    except ValueError as exc:
        return 0.0, {"status": "invalid_schedule", "error": str(exc)}

    local_optima = enumerate_local_optima(scenario)
    best_slots = local_optima.get(agent_id)
    if best_slots is None:
        return 0.0, {"status": "unknown_agent", "agent_id": agent_id}

    matches = [slot in best_slots[day_idx] for day_idx, slot in enumerate(schedule)]
    score = sum(matches) / scenario.num_days if scenario.num_days else 0.0
    detail = {
        "status": "ok" if score >= 1.0 else "partial",
        "agent_id": agent_id,
        "schedule": schedule,
        "best": best_slots,
        "matches": matches,
    }
    return score, detail


def evaluate_global_agent_policy_script(
    code: str,
    *,
    scenario: EVScenario,
    scenario_dir: Path,
    agent_id: int,
    output_filename: str = "global_policy_output.json",
) -> Tuple[float, Dict[str, object]]:
    """Execute and score a coordination policy for a single agent."""

    agent_dir = scenario_dir / "global" / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    script_path = agent_dir / "_candidate_global.py"
    script_path.write_text(code, encoding="utf-8")

    result = _run_python_script(script_path, agent_dir)
    if result.returncode != 0:
        return 0.0, {"status": "error", "stderr": result.stderr}

    output_path = agent_dir / output_filename
    if not output_path.exists():
        return 0.0, {"status": "missing_output"}

    with output_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            return 0.0, {"status": "invalid_json", "error": str(exc)}

    try:
        schedule = _coerce_schedule(payload, num_days=scenario.num_days)
    except ValueError as exc:
        return 0.0, {"status": "invalid_schedule", "error": str(exc)}

    best_allocation, best_score = enumerate_global_optimum(scenario)
    if agent_id < 1 or agent_id > len(scenario.agents):
        return 0.0, {"status": "unknown_agent", "agent_id": agent_id}

    target_schedule = [int(day[agent_id - 1]) for day in best_allocation]
    matches = [slot == target for slot, target in zip(schedule, target_schedule)]
    score = sum(matches) / scenario.num_days if scenario.num_days else 0.0
    detail = {
        "status": "ok" if score >= 1.0 else "partial",
        "agent_id": agent_id,
        "schedule": schedule,
        "target": target_schedule,
        "matches": matches,
        "objective": best_score,
    }
    return score, detail


def evaluate_agent_nudge_response(
    message: str,
    *,
    scenario: EVScenario,
    agent_id: int,
) -> Tuple[float, Dict[str, object]]:
    """Validate a personalised nudge and ensure the slot matches the coordinated plan."""

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        return 0.0, {"status": "invalid_json", "error": str(exc)}

    required_keys = {"persona", "recommended_slots", "message"}
    if not required_keys.issubset(payload):
        return 0.0, {"status": "missing_keys", "payload": payload}

    try:
        recommended_slots = _coerce_schedule(payload["recommended_slots"], num_days=scenario.num_days)
    except (TypeError, ValueError, KeyError) as exc:
        return 0.0, {"status": "bad_slots", "error": str(exc)}

    persona = str(payload["persona"])
    text = str(payload["message"])

    if agent_id < 1 or agent_id > len(scenario.agents):
        return 0.0, {"status": "unknown_agent", "agent_id": agent_id}

    agent = scenario.agents[agent_id - 1]
    if persona != agent.persona:
        return 0.0, {"status": "persona_mismatch", "expected": agent.persona, "received": persona}

    best_allocation, _ = enumerate_global_optimum(scenario)
    target_slots = [int(day[agent_id - 1]) for day in best_allocation]
    matches = [slot == target for slot, target in zip(recommended_slots, target_slots)]
    score = sum(matches) / scenario.num_days if scenario.num_days else 0.0
    detail = {
        "status": "ok" if score >= 1.0 else "partial",
        "agent_id": agent_id,
        "persona": persona,
        "recommended_slots": recommended_slots,
        "target_slots": target_slots,
        "matches": matches,
        "message": text,
    }
    return score, detail

