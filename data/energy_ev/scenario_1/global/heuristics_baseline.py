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
num_agents = len(scenario["agents"])
num_slots = len(scenario["slots"])


def evaluate_day(day, allocation):
    baseline = day["baseline_load"]
    carbon = day.get("carbon_intensity", scenario["carbon_intensity"])
    slot_loads = [baseline[idx] for idx in range(num_slots)]
    for slot in allocation:
        slot_loads[slot] += 1.0
    overload = max(0.0, max(slot_loads) - capacity)
    carbon_term = sum(carbon[slot] for slot in allocation)
    return alpha * overload + beta * carbon_term


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
