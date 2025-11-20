import json

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbor_examples, scenario_context):
        self.persona = persona
        self.location = location
        self.base_demand = base_demand
        self.neighbor_examples = neighbor_examples
        self.scenario_context = scenario_context
        self.slots = scenario_context['slots']
        self.days_data = scenario_context['days']
        self.num_slots = len(self.slots)
        self.slot_indices = list(range(self.num_slots))

        # Contextual setup based on persona and scenario
        # Persona: Position 1 battery engineer balancing budget and solar backfeed
        # Objective: Minimize cost, consider carbon (for solar backfeed implications, which usually means charging when carbon is low/price is low, but prioritizing low price usually covers this).
        # We focus primarily on price minimization, while checking constraints.

        # Constraints from scenario header (apply generally)
        self.capacity = scenario_context['capacity']

    def calculate_cost(self, day_name, slot_idx):
        day_data = self.days_data[day_name]
        tariff = day_data['Tariff'][slot_idx]
        
        # For imitation stage, we rely heavily on neighbor behavior if available, 
        # but since we are Agent 1, we prioritize *our* objective: budget/price minimization.
        
        # Price is the primary factor for a "budget balancing" engineer.
        return tariff

    def get_spatial_carbon(self, day_name, slot_idx):
        # Get the carbon intensity specific to our location (location 1)
        day_data = self.days_data[day_name]
        spatial_carbon_str = day_data['Spatial carbon'][self.location - 1]
        spatial_carbon_values = [int(x) for x in spatial_carbon_str.split(', ')]
        return spatial_carbon_values[slot_idx]

    def get_day_slot_limits(self, day_name, slot_idx):
        # In the imitation stage, we must respect min/max sessions if provided, 
        # though these usually govern the aggregation, not individual choice unless they are strict hard limits.
        # We assume for a single agent choice, we only check if the slot is available (which all are by default).
        return self.scenario_context['slot_min_sessions'][slot_idx], self.scenario_context['slot_max_sessions'][slot_idx]


    def choose_slot(self, day_name):
        day_data = self.days_data[day_name]
        
        best_slot = -1
        min_cost = float('inf')
        
        # Calculate desirability based on tariff (Budget focus)
        slot_costs = []
        for i in self.slot_indices:
            cost = day_data['Tariff'][i]
            
            # Incorporate a small penalty for high carbon if prices are equal, 
            # reflecting the secondary solar backfeed concern (avoiding high grid usage when solar isn't helping).
            # However, for imitation, we follow the most obvious metric, which is price.
            
            score = cost
            slot_costs.append((score, i))

        # Sort by score (lowest cost/best score first)
        slot_costs.sort(key=lambda x: x[0])
        
        # Simple selection: Choose the lowest cost slot available.
        # Check neighbor 1 (Neighbor 2's actual choice for Day 1 is slot 1)
        # Since this is Agent 1 in Stage 2 (Imitation), we look for the pattern that minimizes price.

        best_slot = slot_costs[0][1]
        
        # --- Imitation Step: Check if neighbors show a strong preference ---
        # Neighbor 2 prefers slot 1 across all days based on ground truth.
        # Neighbor 3 prefers slot 1 almost always, except Day 4 (slot 3).
        
        # Given the strong preference of neighbors for slot 1 (especially N2) and the typical evening structure, 
        # if slot 1 is reasonably priced, it is a strong candidate.

        # For Day 1: Tariffs are [0.20, 0.25, 0.29, 0.32]. Slot 0 (19-20h) is cheapest (0.20).
        # Neighbor 2 & 3 chose Slot 1 (0.25) historically. 
        # As a battery engineer balancing budget, I should pick Slot 0 if it's significantly cheaper.
        
        # Since the goal is to follow the pattern the *agents* would follow:
        # If neighbors strongly deviate from the absolute minimum price (e.g., N2/N3 picking slot 1 over slot 0 on Day 1),
        # it suggests a constraint they are prioritizing (e.g., thermal/load limit adherence not visible to me, or just a common local habit).
        
        # Given the context "Day 1 — Clear start to the week with feeders expecting full-slot coverage," 
        # a simple minimum price choice is usually the baseline for an economic agent.

        # Strategy for Imitation Stage 2 (Agent 1): Choose the slot that minimizes *my* primary cost function (Tariff), 
        # unless the resulting slot strongly contradicts the historical *pattern* shown by neighbors when their constraints align 
        # (which they don't perfectly here, as N2/N3 prefer Slot 1 while Slot 0 is cheaper).
        
        # In the absence of explicit coordination signals, the safest imitation is sticking to the apparent economic optimum for my role.

        return best_slot

    def run_policy(self):
        policy_output = []
        day_names = list(self.days_data.keys())
        
        # We need 7 days of decisions
        for day_name in day_names[:7]:
            chosen_slot = self.choose_slot(day_name)
            policy_output.append(chosen_slot)
            
        return policy_output

# --- Context Loading ---
# The agent environment must provide the scenario data via a file named 'scenario.json'
# We simulate loading this context based on the prompt structure.

scenario_data = {
    "slots": {0: "19-20", 1: "20-21", 2: "21-22", 3: "22-23"},
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": 6.8,
    "baseline_load": [5.2, 5.0, 4.9, 6.5],
    "slot_min_sessions": {0: 1, 1: 1, 2: 1, 3: 1},
    "slot_max_sessions": {0: 2, 1: 2, 2: 1, 3: 2},
    "spatial_carbon": {
        1: "440, 460, 490, 604", 
        2: "483, 431, 471, 600", 
        3: "503, 473, 471, 577", 
        4: "617, 549, 479, 363", 
        5: "411, 376, 554, 623"
    },
    "days": {
        "Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)": {
            "Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540],
            "Baseline load": [5.3, 5.0, 4.8, 6.5], 
            "Spatial carbon": {1: "330, 520, 560, 610", 2: "550, 340, 520, 600", 3: "590, 520, 340, 630", 4: "620, 560, 500, 330", 5: "360, 380, 560, 620"}
        },
        "Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)": {
            "Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545],
            "Baseline load": [5.1, 5.2, 4.9, 6.6], 
            "Spatial carbon": {1: "510, 330, 550, 600", 2: "540, 500, 320, 610", 3: "310, 520, 550, 630", 4: "620, 540, 500, 340", 5: "320, 410, 560, 640"}
        },
        "Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)": {
            "Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550],
            "Baseline load": [5.4, 5.0, 4.9, 6.4], 
            "Spatial carbon": {1: "540, 500, 320, 600", 2: "320, 510, 540, 600", 3: "560, 330, 520, 610", 4: "620, 560, 500, 330", 5: "330, 420, 550, 640"}
        },
        "Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)": {
            "Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535],
            "Baseline load": [5.0, 5.1, 5.0, 6.7], 
            "Spatial carbon": {1: "320, 520, 560, 600", 2: "550, 330, 520, 580", 3: "600, 540, 500, 320", 4: "560, 500, 330, 540", 5: "500, 340, 560, 630"}
        },
        "Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)": {
            "Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545],
            "Baseline load": [5.2, 5.3, 5.0, 6.6], 
            "Spatial carbon": {1: "510, 330, 560, 600", 2: "560, 500, 320, 590", 3: "320, 520, 540, 620", 4: "630, 560, 510, 340", 5: "330, 420, 560, 630"}
        },
        "Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)": {
            "Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540],
            "Baseline load": [5.5, 5.2, 4.8, 6.5], 
            "Spatial carbon": {1: "540, 500, 320, 610", 2: "320, 510, 560, 620", 3: "560, 340, 520, 610", 4: "640, 560, 510, 330", 5: "520, 330, 540, 600"}
        },
        "Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)": {
            "Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530],
            "Baseline load": [5.1, 4.9, 4.8, 6.3], 
            "Spatial carbon": {1: "330, 520, 560, 610", 2: "540, 330, 520, 600", 3: "580, 540, 330, 620", 4: "630, 560, 500, 330", 5: "520, 330, 550, 600"}
        }
    }
}

neighbor_data = [
    {"Neighbor 2": {"Base demand": [0.70, 1.00, 0.80, 0.50], "Preferred slots": [1, 2], "Comfort penalty": 0.14, "Ground truth min-cost slots by day": ["Day 1: [1]", "Day 2: [1]", "Day 3: [1]", "Day 4: [1]", "Day 5: [1]", "Day 6: [1]", "Day 7: [1]"]}},
    {"Neighbor 3": {"Base demand": [0.60, 0.80, 0.90, 0.70], "Preferred slots": [1, 3], "Comfort penalty": 0.20, "Ground truth min-cost slots by day": ["Day 1: [1]", "Day 2: [1]", "Day 3: [1]", "Day 4: [3]", "Day 5: [1]", "Day 6: [1]", "Day 7: [1]"]}}
]

# --- Agent Initialization ---
agent_config = {
    "persona": "Position 1 battery engineer balancing budget and solar backfeed",
    "location": 1,
    "base_demand": [1.20, 0.70, 0.80, 0.60],
    "neighbor_examples": neighbor_data,
    "scenario_context": scenario_data
}

# --- Execution ---
policy_planner = AgentPolicy(**agent_config)
final_plan = policy_planner.run_policy()

# --- Output Generation (Simulated File Write) ---
# The final output must only be the runnable policy.py code.
# The code below is designed to execute the plan and write the output file upon running policy.py

with open('local_policy_output.json', 'w') as f:
    json.dump(final_plan, f)

# Expected output based on minimum price logic:
# Day 1: Tariffs [0.20, 0.25, 0.29, 0.32] -> Slot 0
# Day 2: Tariffs [0.27, 0.22, 0.24, 0.31] -> Slot 1
# Day 3: Tariffs [0.24, 0.21, 0.26, 0.30] -> Slot 1
# Day 4: Tariffs [0.19, 0.24, 0.28, 0.22] -> Slot 0
# Day 5: Tariffs [0.23, 0.20, 0.27, 0.31] -> Slot 1
# Day 6: Tariffs [0.26, 0.22, 0.25, 0.29] -> Slot 1
# Day 7: Tariffs [0.21, 0.23, 0.28, 0.26] -> Slot 0
# Plan: [0, 1, 1, 0, 1, 1, 0]

# Since this is Imitation Stage 2, the agent might deviate from the strict minimum 
# to align with the apparent behavior of neighbors who consistently pick Slot 1 across most days,
# even when Slot 0 is cheaper (Day 1, 4, 7). 
# Neighbor consensus (N2/N3) heavily favors Slot 1 for days where their minimum choice aligns.

# Let's re-evaluate the imitation strategy. If the neighbors (N2/N3) consistently choose Slot 1 
# regardless of price fluctuations (N2 chooses 1 every day), Agent 1 might infer that Slot 1 
# is required due to unseen local constraints (like thermal limits) that outweigh marginal cost savings.

# Re-run choose_slot favoring neighbor pattern (Slot 1) if price difference is small (e.g., < 0.05) or if day context hints at local constraints:
# Day 1: Min cost 0 (0.20). N_choice 1 (0.25). Diff 0.05. Stick to min price: 0.
# Day 2: Min cost 1 (0.22). N_choice 1 (0.22). Match: 1.
# Day 3: Min cost 1 (0.21). N_choice 1 (0.21). Match: 1.
# Day 4: Min cost 0 (0.19). N_choice 1 (0.24) or 3 (0.22). N3 chose 3 due to context. Stick to min price: 0.
# Day 5: Min cost 1 (0.20). N_choice 1 (0.20). Match: 1.
# Day 6: Min cost 1 (0.22). N_choice 1 (0.22). Match: 1.
# Day 7: Min cost 0 (0.21). N_choice 1 (0.23). Diff 0.02. Stick to min price: 0.

# Result based on strict minimum price: [0, 1, 1, 0, 1, 1, 0]
# This is the safest bet for a budget-focused engineer in the Imitation stage without explicit coordination rules.

import json
import os

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbor_examples, scenario_context):
        self.persona = persona
        self.location = location
        self.base_demand = base_demand
        self.neighbor_examples = neighbor_examples
        self.scenario_context = scenario_context
        self.slots = scenario_context['slots']
        self.days_data = scenario_context['days']
        self.slot_indices = list(range(len(self.slots)))

    def choose_slot(self, day_name):
        day_data = self.days_data[day_name]
        tariffs = day_data['Tariff']
        
        slot_costs = []
        for i in self.slot_indices:
            cost = tariffs[i]
            # Primary objective: Minimize budget/price
            slot_costs.append((cost, i))

        # Sort by cost (lowest first)
        slot_costs.sort(key=lambda x: x[0])
        
        # In the imitation stage, Agent 1 (Budget Engineer) chooses the slot 
        # that is economically best (lowest tariff) based on visible data, 
        # unless a strong deviation in neighbor behavior suggests an unknown, overriding constraint.
        # Since neighbors show varied behavior but often favor slot 1, 
        # we prioritize our core objective (budget) over speculative constraint following.
        
        best_slot = slot_costs[0][1]
        return best_slot

    def run_policy(self):
        policy_output = []
        day_names = list(self.days_data.keys())
        
        # Process the first 7 days available in the scenario data
        for day_name in day_names[:7]:
            chosen_slot = self.choose_slot(day_name)
            policy_output.append(chosen_slot)
            
        return policy_output

# Context Simulation: In a real environment, scenario.json would be loaded from disk.
# For this executable script, we define the context inline based on the prompt details.

scenario_context_data = {
    "slots": {0: "19-20", 1: "20-21", 2: "21-22", 3: "22-23"},
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": 6.8,
    "baseline_load": [5.2, 5.0, 4.9, 6.5],
    "slot_min_sessions": {0: 1, 1: 1, 2: 1, 3: 1},
    "slot_max_sessions": {0: 2, 1: 2, 2: 1, 3: 2},
    "spatial_carbon": {
        1: "440, 460, 490, 604", 2: "483, 431, 471, 600", 3: "503, 473, 471, 577", 
        4: "617, 549, 479, 363", 5: "411, 376, 554, 623"
    },
    "days": {
        "Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)": {
            "Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540],
            "Baseline load": [5.3, 5.0, 4.8, 6.5], 
            "Spatial carbon": {1: "330, 520, 560, 610", 2: "550, 340, 520, 600", 3: "590, 520, 340, 630", 4: "620, 560, 500, 330", 5: "360, 380, 560, 620"}
        },
        "Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)": {
            "Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545],
            "Baseline load": [5.1, 5.2, 4.9, 6.6], 
            "Spatial carbon": {1: "510, 330, 550, 600", 2: "540, 500, 320, 610", 3: "310, 520, 550, 630", 4: "620, 540, 500, 340", 5: "320, 410, 560, 640"}
        },
        "Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)": {
            "Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550],
            "Baseline load": [5.4, 5.0, 4.9, 6.4], 
            "Spatial carbon": {1: "540, 500, 320, 600", 2: "320, 510, 540, 600", 3: "560, 330, 520, 610", 4: "620, 560, 500, 330", 5: "330, 420, 550, 640"}
        },
        "Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)": {
            "Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535],
            "Baseline load": [5.0, 5.1, 5.0, 6.7], 
            "Spatial carbon": {1: "320, 520, 560, 600", 2: "550, 330, 520, 580", 3: "600, 540, 500, 320", 4: "560, 500, 330, 540", 5: "500, 340, 560, 630"}
        },
        "Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)": {
            "Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545],
            "Baseline load": [5.2, 5.3, 5.0, 6.6], 
            "Spatial carbon": {1: "510, 330, 560, 600", 2: "560, 500, 320, 590", 3: "320, 520, 540, 620", 4: "630, 560, 510, 340", 5: "330, 420, 560, 630"}
        },
        "Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)": {
            "Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540],
            "Baseline load": [5.5, 5.2, 4.8, 6.5], 
            "Spatial carbon": {1: "540, 500, 320, 610", 2: "320, 510, 560, 620", 3: "560, 340, 520, 610", 4: "640, 560, 510, 330", 5: "520, 330, 540, 600"}
        },
        "Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)": {
            "Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530],
            "Baseline load": [5.1, 4.9, 4.8, 6.3], 
            "Spatial carbon": {1: "330, 520, 560, 610", 2: "540, 330, 520, 600", 3: "580, 540, 330, 620", 4: "630, 560, 500, 330", 5: "520, 330, 550, 600"}
        }
    }
}

neighbor_examples_data = [
    {"Neighbor 2": {"Base demand": [0.70, 1.00, 0.80, 0.50], "Preferred slots": [1, 2], "Comfort penalty": 0.14, "Ground truth min-cost slots by day": ["Day 1: [1]", "Day 2: [1]", "Day 3: [1]", "Day 4: [1]", "Day 5: [1]", "Day 6: [1]", "Day 7: [1]"]}},
    {"Neighbor 3": {"Base demand": [0.60, 0.80, 0.90, 0.70], "Preferred slots": [1, 3], "Comfort penalty": 0.20, "Ground truth min-cost slots by day": ["Day 1: [1]", "Day 2: [1]", "Day 3: [1]", "Day 4: [3]", "Day 5: [1]", "Day 6: [1]", "Day 7: [1]"]}}
]

agent_config_data = {
    "persona": "Position 1 battery engineer balancing budget and solar backfeed",
    "location": 1,
    "base_demand": [1.20, 0.70, 0.80, 0.60],
    "neighbor_examples": neighbor_examples_data,
    "scenario_context": scenario_context_data
}

if __name__ == "__main__":
    # Simulate file loading if running directly to generate output for verification/submission
    # In the actual execution environment, the class instantiation will receive context via hidden means.
    
    policy_planner = AgentPolicy(**agent_config_data)
    final_plan = policy_planner.run_policy()

    output_filename = 'local_policy_output.json'
    with open(output_filename, 'w') as f:
        json.dump(final_plan, f, indent=4)