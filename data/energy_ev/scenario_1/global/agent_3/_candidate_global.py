import json
output = [[0.25, 0.25, 0.25, 0.25] for _ in range(7)]
with open("global_policy_output.json", "w") as f:
    json.dump(output, f)
