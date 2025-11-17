"""Prompt builders for the EV coordination task."""

from __future__ import annotations

from pathlib import Path
from textwrap import indent
from typing import Dict

from .scenario import EVScenario


def _load_template(name: str) -> str:
    template_path = Path(__file__).with_name("templates") / name
    with template_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def build_stage_one_prompt(scenario: EVScenario) -> str:
    template = _load_template("stage1.txt")
    return template.format(**_stage_one_context(scenario))


def build_stage_two_prompt(scenario: EVScenario, agent_id: int) -> str:
    template = _load_template("stage2_local_agent.txt")
    context = _shared_context(scenario)
    agent = _agent_block(scenario, agent_id)
    return template.format(**context, **agent)


def build_stage_two_prompts(scenario: EVScenario) -> Dict[int, str]:
    return {agent.id: build_stage_two_prompt(scenario, agent.id) for agent in scenario.agents}


def build_stage_three_prompt(scenario: EVScenario, agent_id: int) -> str:
    template = _load_template("stage3_global_agent.txt")
    context = _shared_context(scenario)
    agent = _agent_block(scenario, agent_id)
    return template.format(**context, **agent)


def build_stage_three_prompts(scenario: EVScenario) -> Dict[int, str]:
    return {agent.id: build_stage_three_prompt(scenario, agent.id) for agent in scenario.agents}


def build_stage_four_prompt(
    scenario: EVScenario,
    agent_id: int,
    *,
    scenario_dir: Path | None = None,
) -> str:
    template = _load_template("stage4_nudge_agent.txt")
    context = _shared_context(scenario)
    agent = _agent_block(scenario, agent_id)
    policy_context = _policy_context(agent_id, scenario_dir)
    return template.format(**context, **agent, **policy_context)


def build_stage_four_prompts(
    scenario: EVScenario,
    *,
    scenario_dir: Path | None = None,
) -> Dict[int, str]:
    return {
        agent.id: build_stage_four_prompt(scenario, agent.id, scenario_dir=scenario_dir)
        for agent in scenario.agents
    }


def _stage_one_context(scenario: EVScenario) -> Dict[str, str]:
    agent_lines = []
    for agent in scenario.agents:
        neighbor_text = "\n".join(
            f"    - Example with neighbor {ex['neighbor_id']}: slot {ex['action']} ({ex['note']})"
            for ex in agent.neighbor_examples
        )
        if not neighbor_text:
            neighbor_text = "    - No exemplar context provided"
        agent_lines.append(
            "\n".join(
                [
                    f"Agent {agent.id} ({agent.persona})",
                    f"  Base demand: {list(agent.base_demand)}",
                    f"  Preferred slots: {list(agent.preferred_slots)}",
                    f"  Comfort penalty: {agent.comfort_penalty}",
                    "  Neighbor ICL:",
                    neighbor_text,
                ]
            )
        )

    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "slots": ", ".join(
            f"{idx}: {label}" for idx, label in enumerate(scenario.slots)
        ),
        "price": ", ".join(f"{val:.2f}" for val in scenario.price),
        "carbon": ", ".join(f"{val:.0f}" for val in scenario.carbon_intensity),
        "baseline": ", ".join(f"{val:.1f}" for val in scenario.baseline_load),
        "capacity": f"{scenario.capacity:.1f}",
        "alpha": f"{scenario.alpha:.2f}",
        "beta": f"{scenario.beta:.2f}",
        "agents": "\n\n".join(agent_lines),
        "daily_conditions": _daily_conditions_block(scenario),
    }


def _shared_context(scenario: EVScenario) -> Dict[str, str]:
    return {
        "scenario_id": scenario.scenario_id,
        "slots": ", ".join(
            f"{idx}: {label}" for idx, label in enumerate(scenario.slots)
        ),
        "price": ", ".join(f"{val:.2f}" for val in scenario.price),
        "carbon": ", ".join(f"{val:.0f}" for val in scenario.carbon_intensity),
        "baseline": ", ".join(f"{val:.1f}" for val in scenario.baseline_load),
        "capacity": f"{scenario.capacity:.1f}",
        "alpha": f"{scenario.alpha:.2f}",
        "beta": f"{scenario.beta:.2f}",
        "daily_conditions": _daily_conditions_block(scenario),
    }


def _agent_block(scenario: EVScenario, agent_id: int) -> Dict[str, str]:
    try:
        agent = next(agent for agent in scenario.agents if agent.id == agent_id)
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown agent id {agent_id}") from exc

    if agent.neighbor_examples:
        neighbor_lines = "\n".join(
            f"- Neighbor {example['neighbor_id']} → slot {example['action']} ({example['note']})"
            for example in agent.neighbor_examples
        )
    else:
        neighbor_lines = "- No neighbour exemplars provided"

    return {
        "agent_id": str(agent.id),
        "persona": agent.persona,
        "base_demand": ", ".join(f"{value:.2f}" for value in agent.base_demand),
        "preferred_slots": ", ".join(str(idx) for idx in agent.preferred_slots) or "None",
        "comfort_penalty": f"{agent.comfort_penalty:.2f}",
        "neighbor_examples": neighbor_lines,
    }


def _policy_context(agent_id: int, scenario_dir: Path | None) -> Dict[str, str]:
    if scenario_dir is None:
        return {
            "local_policy": "    # Local policy not yet available",
            "global_policy": "    # Global policy not yet available",
        }

    local_snippet = _read_policy_snippet(
        scenario_dir / "local" / f"agent_{agent_id}",
        ("best_policy.py", "policy.py", "candidate.py"),
        fallback=f"# Local policy for agent {agent_id} not yet evolved",
    )
    global_snippet = _read_policy_snippet(
        scenario_dir / "global" / f"agent_{agent_id}",
        ("best_policy.py", "policy.py", "candidate.py"),
        fallback=f"# Global policy for agent {agent_id} not yet evolved",
    )

    return {
        "local_policy": indent(local_snippet.strip() or "# Local policy placeholder", "    "),
        "global_policy": indent(global_snippet.strip() or "# Global policy placeholder", "    "),
    }


def _read_policy_snippet(directory: Path, candidates: tuple[str, ...], *, fallback: str) -> str:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return fallback


def _daily_conditions_block(scenario: EVScenario) -> str:
    lines = []
    for idx, day in enumerate(scenario.daily_profiles, start=1):
        note = f" — {day.note}" if day.note else ""
        lines.append(
            "\n".join(
                [
                    f"  Day {idx} ({day.name}{note})",
                    "    Tariff: "
                    + ", ".join(f"{value:.2f}" for value in day.price),
                    "    Carbon: "
                    + ", ".join(f"{value:.0f}" for value in day.carbon_intensity),
                    "    Baseline load: "
                    + ", ".join(f"{value:.1f}" for value in day.baseline_load),
                ]
            )
        )
    if not lines:
        return "  No multi-day context provided"
    return "\n".join(lines)
