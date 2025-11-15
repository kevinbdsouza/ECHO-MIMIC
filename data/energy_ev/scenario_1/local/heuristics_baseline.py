import json
from pathlib import Path

scenario = json.loads(Path("scenario.json").read_text())
price = scenario["price"]
outputs = []

for agent in scenario["agents"]:
    preferred = set(agent.get("preferred_slots", []))
    comfort_penalty = agent.get("comfort_penalty", 0.0)
    costs = []
    for slot_idx, slot_price in enumerate(price):
        penalty = 0.0 if slot_idx in preferred else comfort_penalty
        costs.append((slot_price + penalty, slot_idx))
    costs.sort()
    outputs.append(costs[0][1])

Path("local_policy_output.json").write_text(json.dumps(outputs, indent=2) + "\n")
