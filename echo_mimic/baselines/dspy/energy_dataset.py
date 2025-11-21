"""Shared helpers for loading energy EV baseline data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from echo_mimic.domains.energy_ev.scenario import AgentConfig, EVScenario, load_scenario


def agent_id_from_name(name: str) -> Optional[int]:
    try:
        return int(str(name).split("_")[-1])
    except (TypeError, ValueError):
        return None


def discover_scenarios(root: Path) -> List[Path]:
    """Return scenario directories containing scenario.json under the given root."""
    scenarios: List[Path] = []
    if (root / "scenario.json").exists():
        scenarios.append(root)
    for candidate in sorted(root.glob("scenario_*")):
        if candidate.is_dir() and (candidate / "scenario.json").exists():
            scenarios.append(candidate)
    return scenarios


@lru_cache(maxsize=8)
def load_cached_scenario(path: str) -> EVScenario:
    return load_scenario(Path(path))


def iter_stage_agents(root: Path, stage: str) -> Iterator[Tuple[Path, Path, int, EVScenario, AgentConfig]]:
    """Yield (scenario_dir, agent_dir, agent_id, scenario, agent_config) for a stage directory."""
    for scenario_dir in discover_scenarios(root):
        scenario_path = scenario_dir / "scenario.json"
        if not scenario_path.exists():
            continue
        scenario = load_cached_scenario(str(scenario_path))
        agent_lookup: Dict[int, AgentConfig] = {agent.id: agent for agent in scenario.agents}
        stage_dir = scenario_dir / stage
        if not stage_dir.exists():
            continue
        for agent_dir in sorted(stage_dir.glob("agent_*")):
            agent_id = agent_id_from_name(agent_dir.name)
            if agent_id is None:
                continue
            agent_cfg = agent_lookup.get(agent_id)
            if agent_cfg is None:
                continue
            yield scenario_dir, agent_dir, agent_id, scenario, agent_cfg


def format_agent_context(agent_cfg: AgentConfig) -> str:
    payload = agent_cfg.serialize()
    persona = payload.get("persona", "")
    location = payload.get("location", "")
    preferred = payload.get("preferred_slots", [])
    neighbor_examples = payload.get("neighbor_examples", [])
    parts = [
        f"Persona: {persona}",
        f"Location: {location}",
        f"Preferred slots: {preferred}",
    ]
    if neighbor_examples:
        parts.append("Neighbor exemplars:")
        for idx, example in enumerate(neighbor_examples, start=1):
            parts.append(f"Example {idx}: {json.dumps(example, indent=2, sort_keys=True)}")
    return "\n".join(parts)
