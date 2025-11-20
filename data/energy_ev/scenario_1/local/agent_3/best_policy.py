import json
import numpy as np

# --- Agent Setup ---
SLOTS = [
    {"start": 19, "end": 20, "price": 0.23, "carbon": 700},
    {"start": 20, "end": 21, "price": 0.24, "carbon": 480},
    {"start": 21, "end": 22, "price": 0.27, "carbon": 500},
    {"start": 22, "end": 23, "price": 0.30, "carbon": 750},
]
CAPACITY = 6.8
BASE_LOADS = [5.2, 5.0, 4.9, 6.5]
SLOT_MIN_SESSIONS = [1, 1, 1, 1]
SLOT_MAX_SESSIONS = [2, 2, 1, 2]

# Agent 3 Profile
LOCATION = 3
BASE_DEMAND = [0.60, 0.80, 0.90, 0.70]

# Neighbor Examples for Imitation Context
NEIGHBOR_EXAMPLES = [
    # Neighbor 2 (Location 2, Transformer headroom focus)
    {"location": 2, "base_demand": [0.70, 1.00, 0.80, 0.50], "preferred_slots": [1, 2], "comfort_penalty": 0.14, "ground_truth": [1, 1, 1, 1, 1, 1, 1]},
    # Neighbor 5 (Location 5, Late commuter)
    {"location": 5, "base_demand": [0.50, 0.70, 0.60, 0.90], "preferred_slots": [0, 1], "comfort_penalty": 0.12, "ground_truth": [0, 1, 0, 0, 1, 1, 1]},
]

# --- Scenario Data Loading (Mocked via prompt context) ---

# Define the full scenario data structure based on the prompt
SCENARIO_DATA = {
    "scenario_id": "ev_peak_sharing_1",
    "slots": SLOTS,
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": CAPACITY,
    "baseline_load": BASE_LOADS,
    "slot_min_sessions": SLOT_MIN_SESSIONS,
    "slot_max_sessions": SLOT_MAX_SESSIONS,
    "spatial_carbon": {
        "1": [440, 460, 490, 604], "2": [483, 431, 471, 600], "3": [503, 473, 471, 577],
        "4": [617, 549, 479, 363], "5": [411, 376, 554, 623]
    },
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": {"1": [330, 520, 560, 610], "2": [550, 340, 520, 600], "3": [590, 520, 340, 630], "4": [620, 560, 500, 330], "5": [360, 380, 560, 620]}},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": {"1": [510, 330, 550, 600], "2": [540, 500, 320, 610], "3": [310, 520, 550, 630], "4": [620, 540, 500, 340], "5": [320, 410, 560, 640]}},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": {"1": [540, 500, 320, 600], "2": [320, 510, 540, 600], "3": [560, 330, 520, 610], "4": [620, 560, 500, 330], "5": [330, 420, 550, 640]}},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": {"1": [320, 520, 560, 600], "2": [550, 330, 520, 580], "3": [600, 540, 500, 320], "4": [560, 500, 330, 540], "5": [500, 340, 560, 630]}},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": {"1": [510, 330, 560, 600], "2": [560, 500, 320, 590], "3": [320, 520, 540, 620], "4": [630, 560, 510, 340], "5": [330, 420, 560, 630]}},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": {"1": [540, 500, 320, 610], "2": [320, 510, 560, 620], "3": [560, 340, 520, 610], "4": [640, 560, 510, 330], "5": [520, 330, 540, 600]}},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": {"1": [330, 520, 560, 610], "2": [540, 330, 520, 600], "3": [580, 540, 330, 620], "4": [630, 560, 500, 330], "5": [520, 330, 550, 600]}},
    }
}

# --- Agent 3 Policy Logic (Imitation) ---

class Agent3Policy:
    def __init__(self, scenario_data, base_demand, location):
        self.scenario = scenario_data
        self.base_demand = np.array(base_demand)
        self.location = str(location)
        self.n_slots = len(self.scenario['slots'])
        self.n_days = len(self.scenario['days'])
        
        # Pre-calculate neighbor imitation targets (Ground Truth)
        self.imitation_targets = self._get_imitation_targets()

    def _get_imitation_targets(self):
        """Extract the chosen slot for each day from neighbor examples."""
        targets = []
        day_names = list(self.scenario['days'].keys())
        
        # In imitation stage, the agent strictly follows one of the neighbors' known optimal choices.
        # Agent 3 (Central Ridge, Night Shift) might find Neighbor 2 (L2 analyst, headroom focus)
        # or Neighbor 5 (L5 grad, late commuter) more representative of its own timing needs,
        # or it might simply pick the average/most common choice among them.
        
        # Let's assume Agent 3 observes that Neighbor 5 (late preference, slots 0, 1) and 
        # Neighbor 2 (mid preference, slots 1, 2) represent relevant behaviors.
        
        # Since Agent 3 is a night-shift nurse (base demand peaks late: 0.8, 0.9), 
        # they likely prefer later slots than Neighbor 5 (0, 1) but might align with
        # the comfort/timing concerns of Neighbor 2 (1, 2).
        
        # For strict imitation (Stage 2), we look at the ground truth.
        # We will mimic the neighbor whose behavior seems most aligned with the agent's profile:
        # Agent 3 Base Demand: [0.60, 0.80, 0.90, 0.70] -> Strongest need in slots 1 (20-21) and 2 (21-22).
        # Neighbor 2 Prefers: [1, 2] -> Strong match.
        # Neighbor 5 Prefers: [0, 1] -> Weaker match.
        
        # We choose Neighbor 2's Ground Truth plan as the imitation target.
        return NEIGHBOR_EXAMPLES[0]["ground_truth"]


    def calculate_cost(self, day_data, slot_idx):
        """Calculate the composite cost for Agent 3 in a given slot."""
        
        # 1. Price Cost (Minimized)
        price_cost = day_data['Tariff'][slot_idx]
        
        # 2. Carbon Cost (Minimized, weighted by spatial carbon)
        spatial_carbon_map = day_data['Spatial carbon']
        agent_spatial_carbon = spatial_carbon_map[self.location][slot_idx]
        
        # Use the day's overall carbon intensity as a baseline multiplier for relative weighting
        carbon_factor = day_data['Carbon'][slot_idx] / self.scenario['carbon_intensity'][slot_idx]
        carbon_cost = day_data['Carbon'][slot_idx] + (agent_spatial_carbon * carbon_factor * 0.5) # Weight spatial slightly lower
        
        # 3. Demand/Comfort Cost (Minimized based on agent's profile mismatch)
        agent_demand = self.base_demand[slot_idx]
        
        # Agent 3 is a night-shift nurse, meaning slots 1 and 2 are crucial (0.8, 0.9 demand). 
        # If demand is high, cost is low (no penalty). If demand is low, penalty is high.
        
        # Simple cost heuristic: Penalize heavily if charging outside peak demand (1 or 2)
        comfort_penalty = 0.0
        if slot_idx == 0:
            comfort_penalty = (1.0 - agent_demand) * 0.15 # High penalty if choosing slot 0 when demand is usually high later
        elif slot_idx == 3:
            comfort_penalty = (1.0 - agent_demand) * 0.10 # Medium penalty if choosing slot 3 when demand is usually lower
        elif slot_idx in [1, 2]:
             comfort_penalty = 0.0 # Low penalty if meeting own peak need
             
        # Total Cost (Minimize price and carbon, penalize comfort mismatch)
        # We prioritize matching the neighbor's historical choice heavily, so the cost function
        # here is secondary to the imitation logic in Stage 2.
        total_cost = price_cost + (carbon_cost / 1000.0) + comfort_penalty
        
        return total_cost

    def choose_slot(self, day_index):
        """
        In Stage 2 (Imitation), the agent chooses the slot historically chosen 
        by the most relevant neighbor for that day.
        """
        return self.imitation_targets[day_index]

    def generate_policy(self):
        
        day_names = list(self.scenario['days'].keys())
        policy_output = []
        
        print(f"Agent 3 (Location {self.location}, Night Shift) - Imitating Neighbor 2's historical plan.")
        
        for i in range(self.n_days):
            day_name = day_names[i]
            
            # Decision based purely on imitation target for Stage 2
            chosen_slot = self.choose_slot(i)
            policy_output.append(chosen_slot)
            
            # --- Verification/Debug (Optional: To show what the agent is *ignoring* for imitation) ---
            day_data = self.scenario['days'][day_name]
            
            slot_costs = [self.calculate_cost(day_data, j) for j in range(self.n_slots)]
            
            # Check constraints (Agent 3 assumes 1 session to satisfy min/max constraints easily)
            # Capacity constraint is handled globally, agent just proposes usage.
            
            print(f"{day_name} (Idx {i+1}): Imitation Slot={chosen_slot}")
            # print(f"    Costs: {[f'{c:.3f}' for c in slot_costs]}")
            # print(f"    Best Cost Slot (Self-Optimization): {np.argmin(slot_costs)}")
            
        return policy_output

# --- Execution ---

# 1. Load Scenario Data (Already mocked above, replacing file read for runnable single script)
# In a real environment, this would be:
# with open('scenario.json', 'r') as f:
#     scenario_data = json.load(f)

scenario_data = SCENARIO_DATA
agent_base_demand = BASE_DEMAND
agent_location = LOCATION

# 2. Choose Slots based on Imitation Stage objective
policy_generator = Agent3Policy(scenario_data, agent_base_demand, agent_location)
seven_day_plan = policy_generator.generate_policy()

# 3. Write local_policy_output.json
output_filename = "local_policy_output.json"
with open(output_filename, 'w') as f:
    json.dump(seven_day_plan, f, indent=4)

# print(f"\nGenerated policy: {seven_day_plan}")
# print(f"Output saved to {output_filename}")
# 4. Policy is saved as policy.py (this file content)

# Final required output is just the Python code.
pass # End of simulation setup

# The content below represents the final file 'policy.py'
# --- policy.py contents ---
import json
import numpy as np

# --- Agent Setup ---
SLOTS = [
    {"start": 19, "end": 20, "price": 0.23, "carbon": 700},
    {"start": 20, "end": 21, "price": 0.24, "carbon": 480},
    {"start": 21, "end": 22, "price": 0.27, "carbon": 500},
    {"start": 22, "end": 23, "price": 0.30, "carbon": 750},
]
CAPACITY = 6.8
BASE_LOADS = [5.2, 5.0, 4.9, 6.5]
SLOT_MIN_SESSIONS = [1, 1, 1, 1]
SLOT_MAX_SESSIONS = [2, 2, 1, 2]

# Agent 3 Profile
LOCATION = 3
BASE_DEMAND = [0.60, 0.80, 0.90, 0.70]

# Neighbor Examples for Imitation Context
NEIGHBOR_EXAMPLES = [
    # Neighbor 2 (Location 2, Transformer headroom focus) -> Chosen for imitation match
    {"location": 2, "base_demand": [0.70, 1.00, 0.80, 0.50], "preferred_slots": [1, 2], "comfort_penalty": 0.14, "ground_truth": [1, 1, 1, 1, 1, 1, 1]},
    # Neighbor 5 (Location 5, Late commuter)
    {"location": 5, "base_demand": [0.50, 0.70, 0.60, 0.90], "preferred_slots": [0, 1], "comfort_penalty": 0.12, "ground_truth": [0, 1, 0, 0, 1, 1, 1]},
]

# --- Scenario Data Loading (Mocked via prompt context) ---
SCENARIO_DATA = {
    "scenario_id": "ev_peak_sharing_1",
    "slots": SLOTS,
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": CAPACITY,
    "baseline_load": BASE_LOADS,
    "slot_min_sessions": SLOT_MIN_SESSIONS,
    "slot_max_sessions": SLOT_MAX_SESSIONS,
    "spatial_carbon": {
        "1": [440, 460, 490, 604], "2": [483, 431, 471, 600], "3": [503, 473, 471, 577],
        "4": [617, 549, 479, 363], "5": [411, 376, 554, 623]
    },
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": {"1": [330, 520, 560, 610], "2": [550, 340, 520, 600], "3": [590, 520, 340, 630], "4": [620, 560, 500, 330], "5": [360, 380, 560, 620]}},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": {"1": [510, 330, 550, 600], "2": [540, 500, 320, 610], "3": [310, 520, 550, 630], "4": [620, 540, 500, 340], "5": [320, 410, 560, 640]}},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": {"1": [540, 500, 320, 600], "2": [320, 510, 540, 600], "3": [560, 330, 520, 610], "4": [620, 560, 500, 330], "5": [330, 420, 550, 640]}},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": {"1": [320, 520, 560, 600], "2": [550, 330, 520, 580], "3": [600, 540, 500, 320], "4": [560, 500, 330, 540], "5": [500, 340, 560, 630]}},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": {"1": [510, 330, 560, 600], "2": [560, 500, 320, 590], "3": [320, 520, 540, 620], "4": [630, 560, 510, 340], "5": [330, 420, 560, 630]}},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": {"1": [540, 500, 320, 610], "2": [320, 510, 560, 620], "3": [560, 340, 520, 610], "4": [640, 560, 510, 330], "5": [520, 330, 540, 600]}},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": {"1": [330, 520, 560, 610], "2": [540, 330, 520, 600], "3": [580, 540, 330, 620], "4": [630, 560, 500, 330], "5": [520, 330, 550, 600]}},
    }
}

# --- Agent 3 Policy Logic (Imitation) ---

class Agent3Policy:
    def __init__(self, scenario_data, base_demand, location):
        self.scenario = scenario_data
        self.base_demand = np.array(base_demand)
        self.location = str(location)
        self.n_slots = len(self.scenario['slots'])
        self.n_days = len(self.scenario['days'])
        
        # Pre-calculate neighbor imitation targets (Ground Truth)
        self.imitation_targets = self._get_imitation_targets()

    def _get_imitation_targets(self):
        """Extract the chosen slot for each day from neighbor examples."""
        
        # Agent 3 (Location 3, Night Shift, high demand in slots 1, 2) aligns best 
        # behaviorally with Neighbor 2 (Loc 2, Prefers 1, 2). We use N2's ground truth.
        return NEIGHBOR_EXAMPLES[0]["ground_truth"]

    def calculate_cost(self, day_data, slot_idx):
        """Calculate the composite cost for Agent 3 in a given slot (not strictly used in Stage 2 imitation)."""
        
        price_cost = day_data['Tariff'][slot_idx]
        
        spatial_carbon_map = day_data['Spatial carbon']
        agent_spatial_carbon = spatial_carbon_map[self.location][slot_idx]
        carbon_factor = day_data['Carbon'][slot_idx] / self.scenario['carbon_intensity'][slot_idx]
        carbon_cost = day_data['Carbon'][slot_idx] + (agent_spatial_carbon * carbon_factor * 0.5)
        
        agent_demand = self.base_demand[slot_idx]
        comfort_penalty = 0.0
        if slot_idx == 0:
            comfort_penalty = (1.0 - agent_demand) * 0.15
        elif slot_idx == 3:
            comfort_penalty = (1.0 - agent_demand) * 0.10
        elif slot_idx in [1, 2]:
             comfort_penalty = 0.0
             
        total_cost = price_cost + (carbon_cost / 1000.0) + comfort_penalty
        
        return total_cost

    def choose_slot(self, day_index):
        """
        In Stage 2 (Imitation), the agent chooses the slot historically chosen 
        by the most relevant neighbor for that day.
        """
        return self.imitation_targets[day_index]

    def generate_policy(self):
        
        day_names = list(self.scenario['days'].keys())
        policy_output = []
        
        for i in range(self.n_days):
            # Decision based purely on imitation target for Stage 2
            chosen_slot = self.choose_slot(i)
            policy_output.append(chosen_slot)
            
        return policy_output

# --- Execution ---
# Context data (derived from prompt)
scenario_data = SCENARIO_DATA
agent_base_demand = BASE_DEMAND
agent_location = LOCATION

# 2. Choose Slots based on Imitation Stage objective
policy_generator = Agent3Policy(scenario_data, agent_base_demand, agent_location)
seven_day_plan = policy_generator.generate_policy()

# 3. Write local_policy_output.json
output_filename = "local_policy_output.json"
with open(output_filename, 'w') as f:
    json.dump(seven_day_plan, f, indent=4)