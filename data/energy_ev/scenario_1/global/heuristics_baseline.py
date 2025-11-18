import itertools
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
scenario = json.loads((BASE_DIR / "scenario.json").read_text())
days = scenario.get("days") or [
    {
        "baseline_load": scenario["baseline_load"],
        "carbon_intensity": scenario["carbon_intensity"],
    }
]
capacity = scenario["capacity"]
alpha = scenario["weights"]["alpha"]
beta = scenario["weights"]["beta"]
gamma = scenario["weights"].get("gamma", 1.0)
num_agents = len(scenario["agents"])
num_slots = len(scenario["slots"])
slot_minimums = scenario.get("slot_min_sessions", [0 for _ in range(num_slots)])
slot_maximums = scenario.get(
    "slot_max_sessions", [num_agents for _ in range(num_slots)]
)


def _carbon_for(day, slot, agent):
    spatial = day.get("spatial_carbon", {})
    location = agent.get("location")
    if location in spatial:
        return spatial[location][slot]
    return day.get("carbon_intensity", scenario["carbon_intensity"])[slot]


def evaluate_day(day, allocation):
    baseline = day["baseline_load"]
    slot_loads = [baseline[idx] for idx in range(num_slots)]
    slot_counts = [0 for _ in range(num_slots)]
    carbon_term = 0.0
    for agent_idx, slot in enumerate(allocation):
        slot_loads[slot] += 1.0
        slot_counts[slot] += 1
        carbon_term += _carbon_for(day, slot, scenario["agents"][agent_idx])
    overload = max(0.0, max(slot_loads) - capacity)
    min_penalty = sum(
        max(0, slot_minimums[idx] - slot_counts[idx]) for idx in range(num_slots)
    )
    max_penalty = sum(
        max(0, slot_counts[idx] - slot_maximums[idx]) for idx in range(num_slots)
    )
    return alpha * overload + beta * carbon_term + gamma * (min_penalty + max_penalty)


daily_allocations = []
for day in days:
    best_allocation = None
    best_score = None
    for allocation in itertools.product(range(num_slots), repeat=num_agents):
        score = evaluate_day(day, allocation)
        if best_score is None or score < best_score:
            best_score = score
            best_allocation = allocation
    daily_allocations.append(list(best_allocation))

agent_major = [
    [day_alloc[agent_idx] for day_alloc in daily_allocations]
    for agent_idx in range(num_agents)
]

(BASE_DIR / "global_policy_output.json").write_text(
    json.dumps(agent_major, indent=2) + "\n"
)
