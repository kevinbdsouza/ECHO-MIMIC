"""Prompt builders for the EV coordination task."""

from __future__ import annotations

from pathlib import Path
from textwrap import indent
from typing import Dict, Sequence

from .scenario import (
    AgentConfig,
    DayProfile,
    EVScenario,
    enumerate_global_optimum,
    enumerate_local_optima,
)


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
    agent = _agent_block(scenario, agent_id, ground_truth_mode="global")
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
    agent = _agent_block(scenario, agent_id, ground_truth_mode="global")
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
        agent_lines.append(
            "\n".join(
                [
                    f"Agent {agent.id} ({agent.persona})",
                    f"  Base demand: {list(agent.base_demand)}",
                    f"  Preferred slots: {list(agent.preferred_slots)}",
                    f"  Comfort penalty: {agent.comfort_penalty}",
                    "  Neighbor ICL:",
                    _format_neighbor_examples(
                        agent, scenario, indent_level=4, ground_truth_mode="local"
                    ),
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
        "gamma": f"{scenario.gamma:.2f}",
        "slot_minimums": _format_slot_requirements(scenario.slot_min_sessions),
        "slot_maximums": _format_slot_requirements(scenario.slot_max_sessions),
        "spatial_carbon_summary": _spatial_carbon_summary(scenario),
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
        "carbon_intensity": ", ".join(f"{val:.0f}" for val in scenario.carbon_intensity),
        "capacity": f"{scenario.capacity:.1f}",
        "baseline_load": ", ".join(f"{val:.1f}" for val in scenario.baseline_load),
        "alpha": f"{scenario.alpha:.2f}",
        "beta": f"{scenario.beta:.2f}",
        "gamma": f"{scenario.gamma:.2f}",
        "slot_min_sessions": _format_slot_requirements(scenario.slot_min_sessions),
        "slot_max_sessions": _format_slot_requirements(scenario.slot_max_sessions),
        "spatial_carbon": _spatial_carbon_summary(scenario),
        "days": _daily_conditions_block(scenario),
    }


def _agent_block(
    scenario: EVScenario, agent_id: int, *, ground_truth_mode: str = "local"
) -> Dict[str, str]:
    try:
        agent = next(agent for agent in scenario.agents if agent.id == agent_id)
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown agent id {agent_id}") from exc

    return {
        "agent_id": str(agent.id),
        "persona": agent.persona,
        "base_demand": ", ".join(f"{value:.2f}" for value in agent.base_demand),
        "preferred_slots": ", ".join(str(idx) for idx in agent.preferred_slots) or "None",
        "comfort_penalty": f"{agent.comfort_penalty:.2f}",
        "location": agent.location,
        "neighbor_examples": _format_neighbor_examples(
            agent, scenario, ground_truth_mode=ground_truth_mode
        ),
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
                    _format_spatial_carbon_line(day),
                ]
            )
        )
    if not lines:
        return "  No multi-day context provided"
    return "\n".join(lines)


def _format_slot_requirements(values: Sequence[int]) -> str:
    return ", ".join(f"{idx}: {value}" for idx, value in enumerate(values))


def _spatial_carbon_summary(scenario: EVScenario) -> str:
    zones = sorted({agent.location for agent in scenario.agents})
    if not zones:
        return "uniform"

    num_slots = len(scenario.slots)
    summaries = []
    for zone in zones:
        totals = [0.0 for _ in range(num_slots)]
        count = 0
        for day in scenario.daily_profiles:
            values = day.spatial_carbon.get(zone)
            if values is None:
                continue
            totals = [a + b for a, b in zip(totals, values)]
            count += 1
        if count:
            averages = [value / count for value in totals]
        else:
            averages = list(scenario.carbon_intensity)
        summaries.append(
            f"{zone}: " + ", ".join(f"{value:.0f}" for value in averages)
        )
    return " | ".join(summaries)


def _format_spatial_carbon_line(day: DayProfile) -> str:
    if not day.spatial_carbon:
        return "    Spatial carbon: uniform"
    zone_bits = "; ".join(
        f"{zone}: " + ", ".join(f"{value:.0f}" for value in values)
        for zone, values in sorted(day.spatial_carbon.items())
    )
    return "    Spatial carbon: " + zone_bits


def _format_neighbor_examples(
    agent: AgentConfig,
    scenario: EVScenario,
    *,
    indent_level: int = 0,
    ground_truth_mode: str = "local",
) -> str:
    if ground_truth_mode not in {"local", "global"}:
        raise ValueError("ground_truth_mode must be 'local' or 'global'")

    agent_map = {neighbor.id: neighbor for neighbor in scenario.agents}
    if ground_truth_mode == "local":
        ground_truth_map = enumerate_local_optima(scenario)
    else:
        best_allocation, _ = enumerate_global_optimum(scenario)
        ground_truth_map = {
            agent.id: [[int(day[idx])] for day in best_allocation]
            for idx, agent in enumerate(scenario.agents)
        }

    blocks = []
    for example in agent.neighbor_examples:
        neighbor_id = int(example.get("neighbor_id", -1))
        neighbor = agent_map.get(neighbor_id)

        persona = neighbor.persona if neighbor else str(example.get("persona", "")).strip()
        location = neighbor.location if neighbor else str(example.get("location", "")).strip()
        base_demand = neighbor.base_demand if neighbor else example.get("base_demand")
        preferred = neighbor.preferred_slots if neighbor else example.get("preferred_slots", [])
        comfort = neighbor.comfort_penalty if neighbor else example.get("comfort_penalty")

        ground_truth = ground_truth_map.get(neighbor_id)

        header = f"Neighbor {neighbor_id}"
        if persona:
            header += f" — {persona}"
        if location:
            header += f" (location {location})"

        base_demand_text = (
            ", ".join(f"{float(value):.2f}" for value in base_demand)
            if base_demand is not None
            else "unknown"
        )
        preferred_text = ", ".join(str(int(idx)) for idx in preferred) if preferred else "None"
        comfort_text = f"{float(comfort):.2f}" if comfort is not None else "unknown"
        if ground_truth:
            ground_truth_text = "; ".join(
                f"Day {idx + 1}: {slots}"
                for idx, slots in enumerate(ground_truth)
            )
        else:
            ground_truth_text = "Unavailable"

        lines = [
            f"- {header}",
            f"  Base demand: {base_demand_text}",
            f"  Preferred slots: {preferred_text} | Comfort penalty: {comfort_text}",
            f"  Ground truth min-cost slots by day: {ground_truth_text}",
        ]
        block = "\n".join(lines)
        if indent_level:
            block = indent(block, " " * indent_level)
        blocks.append(block)

    if not blocks:
        return (" " * indent_level) + "- No neighbour exemplars provided" if indent_level else "- No neighbour exemplars provided"

    return "\n".join(blocks)
