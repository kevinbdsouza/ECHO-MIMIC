import json
import numpy as np

# --- Scenario Data Extraction ---
# The scenario data must be hardcoded as per the requirement to run without external loading beyond the agent directory.
SCENARIO_DATA = {
    "scenario_id": "ev_peak_sharing_1",
    "slots": [19, 20, 21, 22],
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": 6.8,
    "baseline_load": [5.2, 5.0, 4.9, 6.5],
    "slot_min_sessions": [1, 1, 1, 1],
    "slot_max_sessions": [2, 2, 1, 2],
    "spatial_carbon": {
        1: [440, 460, 490, 604], 2: [483, 431, 471, 600], 3: [503, 473, 471, 577],
        4: [617, 549, 479, 363], 5: [411, 376, 554, 623]
    },
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": {1: [330, 520, 560, 610], 2: [550, 340, 520, 600], 3: [590, 520, 340, 630], 4: [620, 560, 500, 330], 5: [360, 380, 560, 620]}},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": {1: [510, 330, 550, 600], 2: [540, 500, 320, 610], 3: [310, 520, 550, 630], 4: [620, 540, 500, 340], 5: [320, 410, 560, 640]}},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": {1: [540, 500, 320, 600], 2: [320, 510, 540, 600], 3: [560, 330, 520, 610], 4: [620, 560, 500, 330], 5: [330, 420, 550, 640]}},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": {1: [320, 520, 560, 600], 2: [550, 330, 520, 580], 3: [600, 540, 500, 320], 4: [560, 500, 330, 540], 5: [500, 340, 560, 630]}},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": {1: [510, 330, 560, 600], 2: [560, 500, 320, 590], 3: [320, 520, 540, 620], 4: [630, 560, 510, 340], 5: [330, 420, 560, 630]}},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": {1: [540, 500, 320, 610], 2: [320, 510, 560, 620], 3: [560, 340, 520, 610], 4: [640, 560, 510, 330], 5: [520, 330, 540, 600]}},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": {1: [330, 520, 560, 610], 2: [540, 330, 520, 600], 3: [580, 540, 330, 620], 4: [630, 560, 500, 330], 5: [520, 330, 550, 600]}},
    }
}

# Agent Persona & Constraints
PERSONA = {
    "location": 2,
    "base_demand": [0.70, 1.00, 0.80, 0.50],
}

# Neighbor information (Used only for context/imitation in advanced models, but kept here for structure)
NEIGHBORS = [
    {"location": 1, "base_demand": [1.20, 0.70, 0.80, 0.60], "comfort_penalty": 0.18, "preferred_slots": [0, 2]},
    {"location": 4, "base_demand": [0.90, 0.60, 0.70, 0.80], "comfort_penalty": 0.16, "preferred_slots": [0, 3]},
]

# --- Agent Logic ---

class FeederAnalystPolicy:
    def __init__(self, scenario, persona, neighbors):
        self.scenario = scenario
        self.persona = persona
        self.neighbors = neighbors
        self.location = persona["location"]
        self.base_demand = np.array(persona["base_demand"])
        self.num_slots = len(scenario["price"])
        self.days_of_week = list(scenario["days"].keys())

    def calculate_day_usage(self, day_name, day_data):
        
        # --- Primary Goal: Transformer Headroom (Location 2) ---
        # Prioritize charging when local spatial carbon (proxy for feeder/transformer stress) is low.
        spatial_carbon_day = np.array(day_data["Spatial carbon"][self.location])
        
        max_sc = np.max(spatial_carbon_day)
        # Stress score: 0 (low stress) to 1 (high stress)
        stress_score = spatial_carbon_day / max_sc if max_sc > 0 else np.zeros_like(spatial_carbon_day)
        
        # Headroom Priority: High when stress is low. Weight this highly (70%).
        headroom_priority = 1.0 - stress_score

        # --- Secondary Goal: Cost/Carbon Minimization ---
        tariff = np.array(day_data["Tariff"])
        carbon = np.array(day_data["Carbon"])
        
        # Combine cost and carbon into a single cost measure (normalized appropriately)
        # Scale carbon relative to tariff (e.g., 1 unit of carbon ~ 0.001 of the tariff scale)
        combined_cost = (tariff + carbon / 1000.0) 
        
        # Normalize cost to [0, 1] then invert for preference score
        min_cost = np.min(combined_cost)
        max_cost = np.max(combined_cost)
        
        if max_cost > min_cost:
            normalized_cost = (combined_cost - min_cost) / (max_cost - min_cost)
        else:
            normalized_cost = np.zeros_like(combined_cost)
            
        cost_preference = 1.0 - normalized_cost # High preference for low cost/carbon slots
        
        # --- Combine Priorities ---
        # Blend headroom (70%) and cost/carbon (30%)
        utility = (0.7 * headroom_priority) + (0.3 * cost_preference)
        
        # --- Anchoring to Personal Demand Profile ---
        # We use the agent's base demand profile to anchor the overall scale of usage, 
        # ensuring slot 1 (1.00 demand) is generally higher than slot 3 (0.50 demand) if all else is equal.
        target_shape = self.base_demand / np.max(self.base_demand)
        
        # Final usage is a weighted blend: 70% driven by immediate utility (stress/cost), 30% by inherent demand shape.
        final_usage = np.clip( (0.7 * utility) + (0.3 * target_shape), 0.0, 1.0)

        # --- Enforce Slot Constraints (Soft interpretation of min/max sessions) ---
        for i in range(self.num_slots):
            # Ensure mandatory slots (min_sessions >= 1) have non-zero usage if possible
            if self.scenario["slot_min_sessions"][i] >= 1 and final_usage[i] < 0.1:
                final_usage[i] = 0.15 
            
            # Explicitly respect max session limit (cap usage at 1.0)
            final_usage[i] = np.clip(final_usage[i], 0.0, 1.0)

        return np.round(final_usage, 3).tolist()

    def generate_policy(self):
        policy_output = []
        
        for day_name in self.days_of_week:
            day_data = self.scenario["days"][day_name]
            usage_vector = self.calculate_day_usage(day_name, day_data)
            policy_output.append(usage_vector)
            
        return policy_output

# --- Execution ---
if __name__ == "__main__":
    policy_generator = FeederAnalystPolicy(SCENARIO_DATA, PERSONA, NEIGHBORS)
    usage_plan = policy_generator.generate_policy()
    
    # Output required format
    with open("local_policy_output.json", "w") as f:
        json.dump(usage_plan, f, indent=4)