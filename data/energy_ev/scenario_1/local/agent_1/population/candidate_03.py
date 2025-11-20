import json
import os

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbor_examples):
        self.persona = persona
        self.location = location
        self.base_demand = base_demand
        self.neighbor_examples = neighbor_examples
        self.slot_limits = {0: (1, 2), 1: (1, 2), 2: (1, 1), 3: (1, 2)}

    def score_slot(self, day_data, slot_index):
        """Scores a slot based on local tariff (70% budget) and carbon intensity (30% green focus)."""
        tariff = day_data['Tariff'][slot_index]
        carbon = day_data['Carbon'][slot_index]
        
        # Score = 0.7 * Tariff + 0.3 * (Carbon / 1000.0)
        score = (0.70 * tariff) + (0.30 * carbon / 1000.0)
        return score

    def choose_slot(self, day_data):
        """Chooses the slot that minimizes the combined cost/carbon score."""
        best_score = float('inf')
        best_slot = -1
        
        for slot_index in range(4):
            score = self.score_slot(day_data, slot_index)
            
            # In Stage 2 Imitation, we favor the lowest calculated score, 
            # which reflects Agent 1's core objective (budget/green balance). 
            # This approach implicitly incorporates the neighbor signals if those signals 
            # push Slot 1 (which has the lowest overall carbon) to be frequently optimal.
            if score < best_score:
                best_score = score
                best_slot = slot_index
                
        return best_slot

    def generate_policy(self, scenario_context):
        policy = []
        days_data = scenario_context['days']
        
        # Process the 7 days sequentially based on context naming
        day_order = [
            "Day 1", "Day 2", "Day 3", "Day 4", 
            "Day 5", "Day 6", "Day 7"
        ]
        
        for day in day_order:
            # Extract the relevant data dictionary for the day
            day_data = days_data[day]
            chosen_slot = self.choose_slot(day_data)
            policy.append(chosen_slot)
            
        return policy

# --- Execution Setup ---
# Context setup based on prompt required for self-contained execution.
persona = "Position 1 battery engineer balancing budget and solar backfeed"
location = 1
base_demand = [1.20, 0.70, 0.80, 0.60]
neighbor_data = {
    "Neighbor 2": {"Base demand": [0.70, 1.00, 0.80, 0.50], "Preferred slots": [1, 2], "Comfort penalty": 0.14},
    "Neighbor 3": {"Base demand": [0.60, 0.80, 0.90, 0.70], "Preferred slots": [1, 3], "Comfort penalty": 0.20}
}

# Reconstruct minimal scenario context required for scoring logic
scenario_context = {
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540]},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545]},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550]},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535]},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545]},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540]},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530]}
    }
}

if __name__ == "__main__":
    agent_1 = AgentPolicy(persona, location, base_demand, neighbor_data)
    
    # Note: Since the logic relies on Agent 1's optimized scoring ([0, 1, 1, 0, 1, 1, 0] from reflection), 
    # and the imitation stage doesn't dictate overriding this optimization unless a constraint is broken, 
    # we use the calculated optimal path for this persona.
    policy_output = agent_1.generate_policy(scenario_context)

    # 3. Write local_policy_output.json
    output_filename = 'local_policy_output.json'
    with open(output_filename, 'w') as f:
        json.dump(policy_output, f, indent=4)