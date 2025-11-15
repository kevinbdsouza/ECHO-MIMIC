"""Scenario utilities for the carbon-aware EV charging task."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a single household EV charger."""

    id: int
    base_demand: Tuple[float, float, float, float]
    preferred_slots: Tuple[int, ...]
    comfort_penalty: float
    persona: str
    neighbor_examples: Tuple[Dict[str, object], ...]

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "AgentConfig":
        return AgentConfig(
            id=int(payload["id"]),
            base_demand=tuple(float(x) for x in payload["base_demand"]),
            preferred_slots=tuple(int(x) for x in payload.get("preferred_slots", [])),
            comfort_penalty=float(payload.get("comfort_penalty", 0.0)),
            persona=str(payload.get("persona", "")),
            neighbor_examples=tuple(dict(example) for example in payload.get("neighbor_examples", [])),
        )

    def serialize(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "base_demand": list(self.base_demand),
            "preferred_slots": list(self.preferred_slots),
            "comfort_penalty": self.comfort_penalty,
            "persona": self.persona,
            "neighbor_examples": [dict(example) for example in self.neighbor_examples],
        }


@dataclass(frozen=True)
class EVScenario:
    """Complete configuration for a four-slot EV charging coordination instance."""

    scenario_id: str
    description: str
    slots: Tuple[str, str, str, str]
    capacity: float
    baseline_load: Tuple[float, float, float, float]
    price: Tuple[float, float, float, float]
    carbon_intensity: Tuple[float, float, float, float]
    agents: Tuple[AgentConfig, AgentConfig, AgentConfig, AgentConfig, AgentConfig]
    alpha: float
    beta: float

    def serialize(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "slots": list(self.slots),
            "capacity": self.capacity,
            "baseline_load": list(self.baseline_load),
            "price": list(self.price),
            "carbon_intensity": list(self.carbon_intensity),
            "agents": [agent.serialize() for agent in self.agents],
            "weights": {"alpha": self.alpha, "beta": self.beta},
        }

    @property
    def num_agents(self) -> int:
        return len(self.agents)


def load_scenario(path: Path) -> EVScenario:
    """Load a scenario definition from ``scenario.json``."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    agents = tuple(AgentConfig.from_dict(agent) for agent in payload["agents"])
    weights = payload.get("weights", {})

    return EVScenario(
        scenario_id=str(payload["scenario_id"]),
        description=str(payload.get("description", "")),
        slots=tuple(str(slot) for slot in payload["slots"]),
        capacity=float(payload["capacity"]),
        baseline_load=tuple(float(x) for x in payload["baseline_load"]),
        price=tuple(float(x) for x in payload["price"]),
        carbon_intensity=tuple(float(x) for x in payload["carbon_intensity"]),
        agents=agents,  # type: ignore[arg-type]
        alpha=float(weights.get("alpha", 1.0)),
        beta=float(weights.get("beta", 0.0)),
    )


def compute_local_cost(scenario: EVScenario, agent: AgentConfig, slot: int) -> float:
    """Compute the imitation-stage cost for a single agent choosing ``slot``."""

    base_cost = scenario.price[slot]
    comfort_penalty = 0.0 if slot in agent.preferred_slots else agent.comfort_penalty
    return base_cost + comfort_penalty


def compute_global_cost(scenario: EVScenario, allocation: Sequence[int]) -> float:
    """Compute the global objective for a joint allocation."""

    if len(allocation) != scenario.num_agents:
        raise ValueError("Allocation length must equal number of agents.")

    feeder_overload = 0.0
    slot_loads = [scenario.baseline_load[t] for t in range(len(scenario.slots))]

    for agent_idx, slot in enumerate(allocation):
        slot_loads[slot] += 1.0  # normalized EV energy need

    for slot_idx, load in enumerate(slot_loads):
        exceedance = max(0.0, load - scenario.capacity)
        feeder_overload = max(feeder_overload, exceedance)

    carbon_term = sum(
        scenario.carbon_intensity[slot] for slot in allocation
    )

    return scenario.alpha * feeder_overload + scenario.beta * carbon_term


def enumerate_global_optimum(scenario: EVScenario) -> Tuple[List[int], float]:
    """Brute-force the joint allocation that minimizes the global cost."""

    best_allocation: List[int] | None = None
    best_score: float | None = None

    for allocation in itertools.product(range(len(scenario.slots)), repeat=scenario.num_agents):
        score = compute_global_cost(scenario, allocation)
        if best_score is None or score < best_score:
            best_score = score
            best_allocation = list(allocation)

    if best_allocation is None or best_score is None:
        raise RuntimeError("No allocation evaluated when enumerating optimum.")

    return best_allocation, best_score


def enumerate_local_optima(scenario: EVScenario) -> Dict[int, List[int]]:
    """Return the set of minimum-cost slots for each agent."""

    local_optima: Dict[int, List[int]] = {}
    for agent in scenario.agents:
        costs = [compute_local_cost(scenario, agent, slot) for slot in range(len(scenario.slots))]
        min_cost = min(costs)
        best_slots = [slot for slot, cost in enumerate(costs) if abs(cost - min_cost) < 1e-9]
        local_optima[agent.id] = best_slots
    return local_optima


def dump_ground_truth(directory: Path, scenario: EVScenario) -> None:
    """Write local, global, and nudge ground-truth files into ``directory``."""

    local_dir = directory / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_optima = enumerate_local_optima(scenario)
    local_path = local_dir / "ground_truth.json"
    local_payload = {str(agent_id): slots for agent_id, slots in local_optima.items()}
    local_path.write_text(
        json.dumps(local_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    for agent in scenario.agents:
        agent_local_dir = local_dir / f"agent_{agent.id}"
        agent_local_dir.mkdir(parents=True, exist_ok=True)
        agent_local_payload = {
            "agent_id": agent.id,
            "persona": agent.persona,
            "best_slots": local_optima.get(agent.id, []),
        }
        agent_local_path = agent_local_dir / "ground_truth.json"
        agent_local_path.write_text(
            json.dumps(agent_local_payload, indent=2) + "\n", encoding="utf-8"
        )

    global_dir = directory / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    global_opt, global_score = enumerate_global_optimum(scenario)
    global_path = global_dir / "ground_truth.json"
    global_payload = {
        "best_allocation": global_opt,
        "objective": global_score,
        "metadata": {
            "alpha": scenario.alpha,
            "beta": scenario.beta,
            "capacity": scenario.capacity,
            "baseline_load": list(scenario.baseline_load),
        },
    }
    global_path.write_text(json.dumps(global_payload, indent=2) + "\n", encoding="utf-8")

    for idx, agent in enumerate(scenario.agents, start=1):
        agent_global_dir = global_dir / f"agent_{agent.id}"
        agent_global_dir.mkdir(parents=True, exist_ok=True)
        agent_global_payload = {
            "agent_id": agent.id,
            "persona": agent.persona,
            "recommended_slot": int(global_opt[idx - 1]),
            "objective": global_score,
        }
        agent_global_path = agent_global_dir / "ground_truth.json"
        agent_global_path.write_text(
            json.dumps(agent_global_payload, indent=2) + "\n", encoding="utf-8"
        )

    nudge_dir = directory / "nudge"
    nudge_dir.mkdir(parents=True, exist_ok=True)
    recommended_allocation_path = nudge_dir / "recommended_allocation.json"
    recommended_allocation_payload = {
        "allocation": global_opt,
        "objective": global_score,
        "notes": f"Derived from brute-force enumeration of the global objective for scenario {scenario.scenario_id}.",
    }
    recommended_allocation_path.write_text(
        json.dumps(recommended_allocation_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    for idx, agent in enumerate(scenario.agents, start=1):
        agent_nudge_dir = nudge_dir / f"agent_{agent.id}"
        agent_nudge_dir.mkdir(parents=True, exist_ok=True)
        agent_nudge_payload = {
            "agent_id": agent.id,
            "persona": agent.persona,
            "recommended_slot": int(global_opt[idx - 1]),
            "local_best_slots": local_optima.get(agent.id, []),
            "notes": "Nudge messages should reconcile the agent's imitation heuristic with the coordinated recommendation.",
        }
        agent_nudge_path = agent_nudge_dir / "context.json"
        agent_nudge_path.write_text(
            json.dumps(agent_nudge_payload, indent=2) + "\n", encoding="utf-8"
        )

