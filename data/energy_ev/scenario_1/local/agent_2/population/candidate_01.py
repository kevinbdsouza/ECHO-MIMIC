import json
import numpy as np

# --- Scenario Data Extraction ---
# In a real implementation, this would be loaded from scenario.json
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
    "base_demand": [0.70, 1.00, 0.80, 0.50], # Slot 1 (20-21h) is the highest base demand
}

# Neighbor information (For context)
NEIGHBORS = [
    {"location": 1, "base_demand": [1.20, 0.70, 0.80, 0.60], "comfort_penalty": 0.18, "preferred_slots": [0, 2]},
    {"location": 4, "base_demand": [0.90, 0.60, 0.70, 0.80], "comfort_penalty": 0.16, "preferred_slots": [0, 3]},
]

# --- Agent Logic ---

class NovelPolicy:
    def __init__(self, scenario, persona, neighbors):
        self.scenario = scenario
        self.persona = persona
        self.neighbors = neighbors
        self.location = persona["location"]
        self.base_demand = np.array(persona["base_demand"])
        self.num_slots = len(scenario["price"])
        self.days_of_week = list(scenario["days"].keys())

    def calculate_day_usage(self, day_name, day_data):
        
        # Primary Goal: Headroom (Minimize local spatial carbon stress at Location 2)
        spatial_carbon_day = np.array(day_data["Spatial carbon"][self.location])
        
        # Normalize spatial carbon to get stress score (0=low stress, 1=high stress)
        max_sc = np.max(spatial_carbon_day)
        stress_score = spatial_carbon_day / max_sc if max_sc > 0 else np.zeros_like(spatial_carbon_day)
        
        # Headroom Priority: Higher when stress is low (1 - stress_score)
        headroom_priority = 1.0 - stress_score

        # Secondary Goal (Departure from Parents): Aggressively favor the agent's *highest base demand slot* (Slot 1) 
        # unless stressed AND heavily penalize the *highest price/carbon slot* (Slot 3, index 3).
        
        # 1. Base Demand Anchor: Maximize usage in Slot 1 (index 1, base demand 1.00)
        demand_anchor = np.array([0.1, 1.0, 0.1, 0.1]) 
        
        # 2. Price/Carbon Penalty: Heavily penalize slot 3 (highest global price/carbon in forecast)
        # Global Price/Carbon: [0.23, 0.24, 0.27, 0.30] -> Slot 3 is the worst.
        price_penalty = np.array([0.0, 0.0, 0.0, 1.0]) 
        
        # Combine all factors using novel weighting: 
        # 60% Headroom Mitigation, 30% Demand Anchor, 10% Penalty application
        
        # Apply penalty by subtraction (or reducing the score of the penalized slot)
        raw_score = (0.6 * headroom_priority) + (0.3 * demand_anchor)
        
        # If the demand anchor is low (0.1) for a slot, the score is dominated by headroom.
        # If headroom is good, the score will be decent unless price_penalty is 1.0 for that slot.
        
        # Custom modulation: If the slot is highly stressed OR highly penalized, drastically reduce its score.
        is_stressed = stress_score > 0.7
        is_penalized = price_penalty > 0.9
        
        for i in range(self.num_slots):
            if is_stressed[i] or is_penalized[i]:
                # Reduce score significantly if stressed or penalized
                raw_score[i] *= 0.3 
            elif i == 1:
                # Ensure Slot 1 maintains high score due to base demand priority
                raw_score[i] = np.clip(raw_score[i] + 0.2, 0.0, 1.0)

        
        final_usage = np.clip(raw_score, 0.0, 1.0)

        # Ensure minimum required session compliance (0.1 usage floor if min_sessions=1)
        min_sessions = np.array(self.scenario["slot_min_sessions"])
        for i in range(self.num_slots):
            if min_sessions[i] >= 1 and final_usage[i] < 0.1:
                final_usage[i] = 0.15
            
            # Hard cap based on max sessions (interpreted as max expected utilization for this agent)
            if self.scenario["slot_max_sessions"][i] == 1:
                 final_usage[i] = np.clip(final_usage[i], 0.0, 0.95) # Slightly less than 1.0 if slot is limited to 1 session

        return np.round(final_usage, 3).tolist()

    def generate_policy(self):
        policy_output = []
        day_names = list(self.scenario["days"].keys())
        
        for day_name in day_names:
            day_data = self.scenario["days"][day_name]
            usage_vector = self.calculate_day_usage(day_name, day_data)
            policy_output.append(usage_vector)
            
        return policy_output

# --- Execution ---
if __name__ == "__main__":
    # Simulate loading scenario.json by using the hardcoded structure
    
    policy_generator = NovelPolicy(SCENARIO_DATA, PERSONA, NEIGHBORS)
    usage_plan = policy_generator.generate_policy()
    
    # Output required format: local_policy_output.json
    with open("local_policy_output.json", "w") as f:
        json.dump(usage_plan, f, indent=4)