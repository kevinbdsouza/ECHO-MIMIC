"""Prompt builders for the EV coordination task."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .scenario import EVScenario


def _load_template(name: str) -> str:
    template_path = Path(__file__).with_name("templates") / name
    with template_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def build_stage_one_prompt(scenario: EVScenario) -> str:
    template = _load_template("stage1.txt")
    return template.format(**_prompt_context(scenario))


def build_stage_two_prompt(scenario: EVScenario) -> str:
    template = _load_template("stage2_local.txt")
    return template.format(**_prompt_context(scenario))


def build_stage_three_prompt(scenario: EVScenario) -> str:
    template = _load_template("stage3_global.txt")
    return template.format(**_prompt_context(scenario))


def build_stage_four_prompt(scenario: EVScenario) -> str:
    template = _load_template("stage4_nudge.txt")
    return template.format(**_prompt_context(scenario))


def _prompt_context(scenario: EVScenario) -> Dict[str, str]:
    agent_lines = []
    for agent in scenario.agents:
        neighbor_text = "\n".join(
            f"    - Example with neighbor {ex['neighbor_id']}: slot {ex['action']} ({ex['note']})"
            for ex in agent.neighbor_examples
        )
        agent_lines.append(
            f"Agent {agent.id} ({agent.persona})\n"
            f"  Base demand: {list(agent.base_demand)}\n"
            f"  Preferred slots: {list(agent.preferred_slots)}\n"
            f"  Comfort penalty: {agent.comfort_penalty}\n"
            f"  Neighbor ICL:\n{neighbor_text}"
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
    }
