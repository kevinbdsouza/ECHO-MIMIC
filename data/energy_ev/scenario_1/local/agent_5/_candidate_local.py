import json
import os

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbor_examples, scenario_data):
        self.persona = persona
        self.location = location
        self.base_demand = base_demand
        self.neighbor_examples = neighbor_examples
        self.scenario_data = scenario_data
        self.num_slots = len(scenario_data['slots'])
        self.days_info = scenario_data['days']

    def generate_7_day_plan(self):
        day_names = list(self.days_info.keys())
        plan = []

        # Agent 5: Late Commuter (High demand in Slot 3).
        # Imitation Stage 2: Since we are not explicitly told *who* to imitate, 
        # we combine the primary signal (lowest Tariff) with the secondary 
        # preference from the persona (favoring later slots 2 or 3 in ties).

        for day_name in day_names:
            day_data = self.days_info[day_name]
            
            tariffs = day_data['Tariff']
            
            best_slot = -1
            min_cost = float('inf')
            
            for slot_idx in range(self.num_slots):
                cost = tariffs[slot_idx]
                
                if cost < min_cost:
                    min_cost = cost
                    best_slot = slot_idx
                elif cost == min_cost:
                    # Tie-breaking: Prefer later slots (aligns with late commuter persona, 
                    # which is a refinement over pure tariff minimization).
                    if slot_idx > best_slot:
                        best_slot = slot_idx

            plan.append(best_slot)
            
        return plan

# --- Scenario and Profile Data Setup (Simulating loaded context) ---

scenario_context = {
    "slots": [19, 20, 21, 22],
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": 6.8,
    "baseline_load": [5.2, 5.0, 4.9, 6.5],
    "slot_min_sessions": [1, 1, 1, 1],
    "slot_max_sessions": [2, 2, 1, 2],
    "spatial_carbon": {
        "location 1": "440, 460, 490, 604", "location 2": "483, 431, 471, 600",
        "location 3": "503, 473, 471, 577", "location 4": "617, 549, 479, 363",
        "location 5": "411, 376, 554, 623"
    },
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5]},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6]},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4]},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7]},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6]},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5]},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3]}
    }
}

AGENT_PERSONA = "Position 5 graduate tenant commuting late from campus"
AGENT_LOCATION = 5
AGENT_BASE_DEMAND = [0.50, 0.70, 0.60, 0.90]
NEIGHBOR_EXAMPLES = [
    {"Neighbor 4": {"Base demand": [0.90, 0.60, 0.70, 0.80], "Preferred slots": [0, 3], "Comfort penalty": 0.16, "Ground truth min-cost slots by day": ["Day 1: [0]", "Day 2: [0]", "Day 3: [0]", "Day 4: [0]", "Day 5: [0]", "Day 6: [3]", "Day 7: [0]"]}},
    {"Neighbor 1": {"Base demand": [1.20, 0.70, 0.80, 0.60], "Preferred slots": [0, 2], "Comfort penalty": 0.18, "Ground truth min-cost slots by day": ["Day 1: [0]", "Day 2: [2]", "Day 3: [2]", "Day 4: [0]", "Day 5: [0]", "Day 6: [2]", "Day 7: [2]"]}
]

# 2. Choose the slot plan based on the imitation objective (Lowest Tariff, tie-break to later slot)
policy_engine = AgentPolicy(
    persona=AGENT_PERSONA,
    location=AGENT_LOCATION,
    base_demand=AGENT_BASE_DEMAND,
    neighbor_examples=NEIGHBOR_EXAMPLES,
    scenario_data=scenario_context
)

output_plan = policy_engine.generate_7_day_plan()

# 3. Write local_policy_output.json
output_filename = "local_policy_output.json"
with open(output_filename, 'w') as f:
    json.dump(output_plan, f, indent=4)