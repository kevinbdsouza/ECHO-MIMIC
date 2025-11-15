import itertools
import json
from pathlib import Path

scenario = json.loads(Path("scenario.json").read_text())
price = scenario["price"]
carbon = scenario["carbon_intensity"]
baseline = scenario["baseline_load"]
capacity = scenario["capacity"]
alpha = scenario["weights"]["alpha"]
beta = scenario["weights"]["beta"]
num_agents = len(scenario["agents"])
num_slots = len(price)


def evaluate(allocation):
    slot_loads = [baseline[idx] for idx in range(num_slots)]
    for slot in allocation:
        slot_loads[slot] += 1.0
    overload = max(0.0, max(slot_loads) - capacity)
    carbon_term = sum(carbon[slot] for slot in allocation)
    return alpha * overload + beta * carbon_term

best_allocation = None
best_score = None

for allocation in itertools.product(range(num_slots), repeat=num_agents):
    score = evaluate(allocation)
    if best_score is None or score < best_score:
        best_score = score
        best_allocation = allocation

Path("global_policy_output.json").write_text(
    json.dumps(list(best_allocation), indent=2) + "\n"
)
