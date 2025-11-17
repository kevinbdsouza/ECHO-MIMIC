import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
scenario = json.loads((BASE_DIR / "scenario.json").read_text())
days = scenario.get("days") or [
    {
        "price": scenario["price"],
    }
]
outputs = []

for agent in scenario["agents"]:
    preferred = set(agent.get("preferred_slots", []))
    comfort_penalty = agent.get("comfort_penalty", 0.0)
    schedule = []
    for day in days:
        price = day["price"]
        costs = []
        for slot_idx, slot_price in enumerate(price):
            penalty = 0.0 if slot_idx in preferred else comfort_penalty
            costs.append((slot_price + penalty, slot_idx))
        costs.sort()
        schedule.append(costs[0][1])
    outputs.append(schedule)

(BASE_DIR / "local_policy_output.json").write_text(
    json.dumps(outputs, indent=2) + "\n"
)
