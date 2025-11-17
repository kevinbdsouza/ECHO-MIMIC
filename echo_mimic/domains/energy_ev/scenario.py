"""Scenario utilities for the carbon-aware EV charging task."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class DayProfile:
    """Daily context for tariff, baseline, and carbon conditions."""

    name: str
    baseline_load: Tuple[float, float, float, float]
    price: Tuple[float, float, float, float]
    carbon_intensity: Tuple[float, float, float, float]
    note: str = ""

    @staticmethod
    def from_dict(payload: Dict[str, object], *, slots: Sequence[str]) -> "DayProfile":
        def _coerce_sequence(values: Sequence[object], label: str) -> Tuple[float, float, float, float]:
            if len(values) != len(slots):
                raise ValueError(
                    f"{label} for day '{payload.get('name')}' must have {len(slots)} entries"
                )
            return tuple(float(x) for x in values)  # type: ignore[return-value]

        baseline = _coerce_sequence(payload["baseline_load"], "baseline_load")
        price = _coerce_sequence(payload["price"], "price")
        carbon = _coerce_sequence(payload["carbon_intensity"], "carbon_intensity")
        return DayProfile(
            name=str(payload.get("name", "Day")),
            baseline_load=baseline,
            price=price,
            carbon_intensity=carbon,
            note=str(payload.get("note", "")),
        )

    def serialize(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "baseline_load": list(self.baseline_load),
            "price": list(self.price),
            "carbon_intensity": list(self.carbon_intensity),
            "note": self.note,
        }

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
    daily_profiles: Tuple[DayProfile, ...]
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
            "days": [day.serialize() for day in self.daily_profiles],
            "weights": {"alpha": self.alpha, "beta": self.beta},
        }

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def num_days(self) -> int:
        return len(self.daily_profiles)


def load_scenario(path: Path) -> EVScenario:
    """Load a scenario definition from ``scenario.json``."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    slots = tuple(str(slot) for slot in payload["slots"])
    agents = tuple(AgentConfig.from_dict(agent) for agent in payload["agents"])
    weights = payload.get("weights", {})

    days_payload = payload.get("days")
    if days_payload:
        daily_profiles = tuple(
            DayProfile.from_dict(day, slots=slots) for day in days_payload
        )
    else:
        # Backward compatibility: repeat the single-day parameters once.
        fallback = DayProfile(
            name="Day 1",
            baseline_load=tuple(float(x) for x in payload["baseline_load"]),
            price=tuple(float(x) for x in payload["price"]),
            carbon_intensity=tuple(float(x) for x in payload["carbon_intensity"]),
        )
        daily_profiles = (fallback,)

    return EVScenario(
        scenario_id=str(payload["scenario_id"]),
        description=str(payload.get("description", "")),
        slots=slots,
        capacity=float(payload["capacity"]),
        baseline_load=tuple(float(x) for x in payload["baseline_load"]),
        price=tuple(float(x) for x in payload["price"]),
        carbon_intensity=tuple(float(x) for x in payload["carbon_intensity"]),
        daily_profiles=daily_profiles,
        agents=agents,  # type: ignore[arg-type]
        alpha=float(weights.get("alpha", 1.0)),
        beta=float(weights.get("beta", 0.0)),
    )


def compute_local_cost(
    scenario: EVScenario, agent: AgentConfig, day_idx: int, slot: int
) -> float:
    """Compute the imitation-stage cost for a single agent choosing ``slot``."""

    day = scenario.daily_profiles[day_idx]
    base_cost = day.price[slot]
    comfort_penalty = 0.0 if slot in agent.preferred_slots else agent.comfort_penalty
    return base_cost + comfort_penalty


def compute_global_cost(
    scenario: EVScenario, allocation_by_day: Sequence[Sequence[int]]
) -> float:
    """Compute the global objective for a multi-day joint allocation."""

    if len(allocation_by_day) != scenario.num_days:
        raise ValueError("Allocation must provide a schedule for every day.")

    total_cost = 0.0
    for day_idx, allocation in enumerate(allocation_by_day):
        if len(allocation) != scenario.num_agents:
            raise ValueError("Each day must assign slots for every agent.")
        day = scenario.daily_profiles[day_idx]
        slot_loads = [day.baseline_load[t] for t in range(len(scenario.slots))]
        for slot in allocation:
            slot_loads[slot] += 1.0
        feeder_overload = 0.0
        for load in slot_loads:
            exceedance = max(0.0, load - scenario.capacity)
            feeder_overload = max(feeder_overload, exceedance)

        carbon_term = sum(day.carbon_intensity[slot] for slot in allocation)
        total_cost += scenario.alpha * feeder_overload + scenario.beta * carbon_term

    return total_cost


def enumerate_global_optimum(scenario: EVScenario) -> Tuple[List[List[int]], float]:
    """Brute-force the joint allocation that minimizes the global cost per day."""

    best_allocations: List[List[int]] = []
    total_cost = 0.0

    for day_idx in range(scenario.num_days):
        day_best: List[int] | None = None
        day_score: float | None = None
        for allocation in itertools.product(range(len(scenario.slots)), repeat=scenario.num_agents):
            day_cost = _day_global_cost(scenario, day_idx, allocation)
            if day_score is None or day_cost < day_score:
                day_score = day_cost
                day_best = list(allocation)
        if day_best is None or day_score is None:
            raise RuntimeError("Failed to enumerate day-level optimum.")
        best_allocations.append(day_best)
        total_cost += day_score

    return best_allocations, total_cost


def _day_global_cost(
    scenario: EVScenario, day_idx: int, allocation: Sequence[int]
) -> float:
    day = scenario.daily_profiles[day_idx]
    slot_loads = [day.baseline_load[t] for t in range(len(scenario.slots))]
    for slot in allocation:
        slot_loads[slot] += 1.0
    feeder_overload = 0.0
    for load in slot_loads:
        exceedance = max(0.0, load - scenario.capacity)
        feeder_overload = max(feeder_overload, exceedance)
    carbon_term = sum(day.carbon_intensity[slot] for slot in allocation)
    return scenario.alpha * feeder_overload + scenario.beta * carbon_term


def enumerate_local_optima(scenario: EVScenario) -> Dict[int, List[List[int]]]:
    """Return the set of minimum-cost slots for each agent and day."""

    local_optima: Dict[int, List[List[int]]] = {}
    for agent in scenario.agents:
        per_day: List[List[int]] = []
        for day_idx in range(scenario.num_days):
            costs = [
                compute_local_cost(scenario, agent, day_idx, slot)
                for slot in range(len(scenario.slots))
            ]
            min_cost = min(costs)
            best_slots = [
                slot for slot, cost in enumerate(costs) if abs(cost - min_cost) < 1e-9
            ]
            per_day.append(best_slots)
        local_optima[agent.id] = per_day
    return local_optima


def dump_ground_truth(directory: Path, scenario: EVScenario) -> None:
    """Write local, global, and nudge ground-truth files into ``directory``."""

    local_dir = directory / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_optima = enumerate_local_optima(scenario)
    local_path = local_dir / "ground_truth.json"
    local_payload = {
        str(agent_id): slots_by_day for agent_id, slots_by_day in local_optima.items()
    }
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
            "best_slots_by_day": local_optima.get(agent.id, []),
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
            "days": [day.serialize() for day in scenario.daily_profiles],
        },
    }
    global_path.write_text(json.dumps(global_payload, indent=2) + "\n", encoding="utf-8")

    for idx, agent in enumerate(scenario.agents, start=1):
        agent_global_dir = global_dir / f"agent_{agent.id}"
        agent_global_dir.mkdir(parents=True, exist_ok=True)
        agent_global_payload = {
            "agent_id": agent.id,
            "persona": agent.persona,
            "recommended_slots": [int(day[idx - 1]) for day in global_opt],
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
            "recommended_slots": [int(day[idx - 1]) for day in global_opt],
            "local_best_slots_by_day": local_optima.get(agent.id, []),
            "notes": "Nudge messages should reconcile the agent's imitation heuristic with the coordinated recommendation.",
        }
        agent_nudge_path = agent_nudge_dir / "context.json"
        agent_nudge_path.write_text(
            json.dumps(agent_nudge_payload, indent=2) + "\n", encoding="utf-8"
        )

