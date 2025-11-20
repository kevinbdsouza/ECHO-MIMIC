"""Evaluation helpers for EV charging heuristics and nudges."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ...common import (
    CommandOutputCapture,
    build_model,
    ensure_rate_limiter,
    extract_python_code,
    make_code_validator,
    pushd,
)
from ...config import Config
from ...rate_limiter import send_message_with_retry
from .scenario import (
    EVScenario,
    enumerate_global_optimum,
    enumerate_local_optima,
    compute_global_cost,
)

_capture = CommandOutputCapture()
_cfg = Config()
_rate_limiter = ensure_rate_limiter(_cfg)
_AGENT_SYSTEM_INSTRUCTION = (
    "You are an EV owner deciding whether to adjust your personal charging heuristic. "
    "Only rewrite your policy if a received nudge benefits you on any of the things you care about. "
    "Always return a complete, executable Python script that writes local_policy_output.json with seven slot indices."
)
_AGENT_FIX_INSTRUCTION = (
    "You debug Python scripts authored by EV owners reacting to nudges. "
    "Given the code and stderr, return a corrected script that writes seven integers to local_policy_output.json "
    "and performs no other IO or side effects."
)


def _run_python_script(script_path: Path, workdir: Path) -> subprocess.CompletedProcess:
    with pushd(workdir):
        exit_code, stdout, stderr = _capture.run_python_script(script_path.name)
    return subprocess.CompletedProcess(
        args=[script_path.name],
        returncode=exit_code,
        stdout=stdout,
        stderr=stderr,
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


def _scenario_root_candidates() -> List[Path]:
    roots = []
    env_root = os.getenv("ECHO_SCENARIO_ROOT")
    if env_root:
        roots.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (
        repo_root / "data",
        repo_root,
        Path.cwd(),
    ):
        if candidate not in roots:
            roots.append(candidate)
    return [path for path in roots if path.exists()]


@lru_cache(maxsize=4)
def _locate_scenario_dir(scenario_id: str) -> Optional[Path]:
    for root in _scenario_root_candidates():
        for scenario_path in root.rglob("scenario.json"):
            if scenario_path.parent.name.startswith("agent_"):
                continue
            try:
                with scenario_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            if payload.get("scenario_id") == scenario_id:
                return scenario_path.parent
    return None


def _load_agent_local_policy_snippet(agent_id: int, scenario_id: str) -> str:
    scenario_dir = _locate_scenario_dir(scenario_id)
    if scenario_dir is not None:
        agent_dir = scenario_dir / "local" / f"agent_{agent_id}"
        for candidate in ("best_policy.py", "policy.py", "candidate.py"):
            path = agent_dir / candidate
            if path.exists():
                return path.read_text(encoding="utf-8")
    return "# Local policy unavailable; using baseline imitation instead."


def _ensure_agent_workspace(scenario: EVScenario, agent_id: int) -> Tuple[Path, bool]:
    """Return a directory containing scenario.json for executing agent code."""

    scenario_dir = _locate_scenario_dir(scenario.scenario_id)
    if scenario_dir:
        agent_dir = scenario_dir / "nudge" / f"agent_{agent_id}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = agent_dir / "scenario_cp.json"
        if not scenario_path.exists():
            scenario_path.write_text(json.dumps(scenario.serialize(), indent=2) + "\n", encoding="utf-8")
        return agent_dir, False

    temp_dir = Path(tempfile.mkdtemp(prefix="agent_nudge_"))
    scenario_path = temp_dir / "scenario.json"
    scenario_path.write_text(json.dumps(scenario.serialize(), indent=2) + "\n", encoding="utf-8")
    return temp_dir, True


@lru_cache(maxsize=1)
def _agent_model(model_name: Optional[str] = None):
    name = model_name or _cfg.lm
    return build_model(name, _AGENT_SYSTEM_INSTRUCTION)


@lru_cache(maxsize=1)
def _agent_fix_model(model_name: Optional[str] = None):
    name = model_name or _cfg.lm
    return build_model(name, _AGENT_FIX_INSTRUCTION)


@lru_cache(maxsize=8)
def _agent_policy_validator(workdir: Path, model_name: Optional[str] = None):
    return make_code_validator(
        workdir=workdir,
        capture=_capture,
        fix_model=_agent_fix_model(model_name),
        rate_limiter=_rate_limiter,
        default_script="_nudged_policy.py",
        default_attempts=2,
    )


def _completion_text(completion: object) -> str:
    text = getattr(completion, "text", "")
    if text:
        return text
    parts: List[str] = []
    for part in getattr(completion, "parts", []):
        candidate = getattr(part, "text", "")
        if candidate:
            parts.append(candidate)
    return "\n".join(parts)


def _build_agent_prompt(
    *,
    persona: str,
    local_policy: str,
    nudge_message: str,
) -> str:
    return (
        "You are the EV driver described by the persona below.\n"
        "Consider whether the incoming nudge improves your own outcomes. "
        "Use the message content itself to decide how, if at all, to adjust your charging heuristic—do not rely on any hidden recommendations. "
        "Only change your heuristic if it clearly benefits or influences you; otherwise keep your current approach. "
        "Even when you make changes, make only measurable changes to your behaviour and not drastic ones. \n\n"
        f"Persona:\n{persona}\n\n"
        "Your current local heuristic:\n"
        f"{local_policy}\n\n"
        "Nudge message:\n"
        f"{nudge_message}\n\n"
        "Respond with the full Python script you will follow going forward. "
        "The script must write a JSON list of seven integers to local_policy_output.json and perform no other changes to I/O. "
        "Return only Python source code with no fences or narration."
    )


def _request_modified_policy(prompt: str) -> str:
    model = _agent_model()
    session = model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, _rate_limiter)
    text = _completion_text(completion).strip()
    if not text:
        raise RuntimeError("Model produced empty policy update")
    return extract_python_code(text)


def _validate_nudged_policy(code: str, workspace: Path) -> str:
    validator = _agent_policy_validator(workspace)
    return validator(code, script_name="_nudged_policy.py", max_attempts=2)


def _run_agent_policy(code: str, workspace: Path, *, agent_id: int, num_days: int) -> Tuple[Optional[List[int]], Dict[str, object]]:
    script_path = workspace / "_nudged_policy.py"
    script_path.write_text(code, encoding="utf-8")

    output_path = workspace / "local_policy_output.json"
    if output_path.exists():
        output_path.unlink()

    result = _run_python_script(script_path, workspace)
    if result.returncode != 0:
        return None, {"status": "error", "stderr": result.stderr}
    if not output_path.exists():
        return None, {"status": "missing_output"}

    with output_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            return None, {"status": "invalid_json", "error": str(exc)}

    try:
        schedule = _coerce_schedule(payload, num_days=num_days)
    except ValueError as exc:
        return None, {"status": "invalid_schedule", "error": str(exc)}

    return schedule, {
        "status": "ok",
        "script_path": str(script_path),
        "output_path": str(output_path),
    }

def evaluate_local_agent_policy_script(
    code: str,
    *,
    scenario: EVScenario,
    scenario_dir: Path,
    agent_id: int,
    output_filename: str = "local_policy_output.json",
) -> Tuple[float, Dict[str, object]]:
    """Execute and score an imitation policy for a single agent."""

    agent_dir = scenario_dir
    script_path = agent_dir / "_candidate_local.py"
    script_path.write_text(code, encoding="utf-8")

    output_path = agent_dir / output_filename
    # Remove any stale output from previous candidates so each run is evaluated on
    # the code we just wrote.
    if output_path.exists():
        output_path.unlink()

    result = _run_python_script(script_path, agent_dir)
    if result.returncode != 0:
        return 0.0, {"status": "error", "stderr": result.stderr}
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

    agent_dir = scenario_dir
    script_path = agent_dir / "_candidate_global.py"
    script_path.write_text(code, encoding="utf-8")

    output_path = agent_dir / output_filename
    if output_path.exists():
        output_path.unlink()

    result = _run_python_script(script_path, agent_dir)
    if result.returncode != 0:
        return 0.0, {"status": "error", "stderr": result.stderr}
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
    """Simulate an agent's response to a nudge by updating and executing their heuristic."""

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

    best_allocation, best_objective = enumerate_global_optimum(scenario)
    target_slots = [int(day[agent_id - 1]) for day in best_allocation]

    local_policy = _load_agent_local_policy_snippet(agent_id, scenario.scenario_id)
    prompt = _build_agent_prompt(
        persona=persona,
        local_policy=local_policy,
        nudge_message=text,
    )

    try:
        modified_policy = _request_modified_policy(prompt)
    except Exception as exc:
        return 0.0, {
            "status": "llm_error",
            "error": str(exc),
            "persona": persona,
            "agent_id": agent_id,
        }

    workspace, cleanup_needed = _ensure_agent_workspace(scenario, agent_id)

    try:
        try:
            validated_policy = _validate_nudged_policy(modified_policy, workspace)
        except Exception as exc:
            return 0.0, {
                "status": "validation_error",
                "error": str(exc),
                "agent_id": agent_id,
                "persona": persona,
                "recommended_slots": recommended_slots,
                "message": text,
            }
        schedule, run_detail = _run_agent_policy(
            validated_policy,
            workspace,
            agent_id=agent_id,
            num_days=scenario.num_days,
        )
    finally:
        if cleanup_needed:
            shutil.rmtree(workspace, ignore_errors=True)

    if schedule is None:
        run_detail.update(
            {
                "agent_id": agent_id,
                "persona": persona,
                "recommended_slots": recommended_slots,
                "message": text,
            }
        )
        return 0.0, run_detail

    matches = [slot == target for slot, target in zip(schedule, target_slots)]
    score = sum(matches) / scenario.num_days if scenario.num_days else 0.0
    detail = {
        "status": "ok" if score >= 1.0 else "partial",
        "agent_id": agent_id,
        "persona": persona,
        "recommended_slots": recommended_slots,
        "message": text,
        "nudged_schedule": schedule,
        "target_slots": target_slots,
        "matches": matches,
        "objective": best_objective,
    }
    return score, detail
