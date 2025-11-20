import json
import os

class EVPolicy:
    def __init__(self):
        # Parameters from the prompt
        self.alpha = 40.00  # Comfort/Carbon weight
        self.beta = 0.50    # Congestion weight (used implicitly via capacity constraints)
        self.gamma = 12.00  # Neighbor coordination weight

        self.num_slots = 4
        self.capacity = 6.8
        self.base_demand = [0.90, 0.60, 0.70, 0.80]

        self.neighbor_examples = [
            {
                "name": "Neighbor 3",
                "location": 3,
                "base_demand": [0.60, 0.80, 0.90, 0.70],
                "preferred_slots": [1, 3],
                "comfort_penalty": 0.20,
                "ground_truth_min_cost_slots_by_day": [2, 0, 1, 3, 0, 1, 2]
            },
            {
                "name": "Neighbor 5",
                "location": 5,
                "base_demand": [0.50, 0.70, 0.60, 0.90],
                "preferred_slots": [0, 1],
                "comfort_penalty": 0.12,
                "ground_truth_min_cost_slots_by_day": [0, 0, 0, 0, 0, 1, 1]
            }
        ]

        # Load scenario data
        self.scenario = self._load_scenario()
        self.day_keys = list(self.scenario['days'].keys())

    def _load_scenario(self):
        # Assuming scenario.json is in the same directory as policy.py
        try:
            with open("scenario.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: scenario.json not found in the current directory.")
            return None
        except json.JSONDecodeError:
            print("Error: Could not decode scenario.json.")
            return None

    def _get_slot_constraints(self, day_data):
        # Extract slot limits for the current day context
        # In Stage 3, limits are constant for all days unless explicitly overridden (which they aren't here)
        return {
            'min_sessions': day_data['slot_min_sessions'],
            'max_sessions': day_data['slot_max_sessions']
        }

    def _calculate_cost(self, day_index, slot_index, day_data):
        # 1. Comfort/Carbon Cost (Local Goal)
        # Persona: Position 4 retirees guarding comfort. High comfort priority.
        # Base demand: [0.90, 0.60, 0.70, 0.80] (Highest demand in Slot 0 and Slot 3)
        
        # Comfort Penalty (Inverse relationship to utility/cost):
        # Higher base demand suggests the agent strongly prefers *not* to be forced out of these slots.
        # A high demand slot that is also highly preferred gets a lower penalty if used.
        
        # Since the persona prioritizes comfort, we penalize deviation from slots where their base demand is high.
        comfort_preference_score = self.base_demand[slot_index]
        
        # Lower comfort score means higher penalty (cost)
        # We invert the preference: Cost = Max_Pref - Current_Pref
        max_pref = max(self.base_demand)
        comfort_cost = self.alpha * (max_pref - comfort_preference_score)

        # Carbon Cost (Grid Goal): Minimize carbon intensity
        carbon_intensity = day_data['Carbon'][slot_index]
        carbon_cost = carbon_intensity  # Simple direct cost

        # Congestion Cost (Grid Goal): Penalize loading relative to capacity (baseline load factored in)
        # Note: We don't know *other* agents' loads yet, so we use baseline + agent demand relative to capacity
        
        # Agent demand (simplified: using base_demand as a proxy for relative consumption effort)
        agent_load_proxy = self.base_demand[slot_index]
        baseline = day_data['Baseline load'][slot_index]
        total_load_proxy = baseline + agent_load_proxy
        
        # A simple congestion heuristic: penalize slots approaching capacity
        # We scale this using beta, though beta is often used for weighting sessions, we use it here for scaling effect.
        congestion_cost = self.beta * (total_load_proxy / self.capacity)

        # 2. Neighbor Coordination Cost (Collective Goal)
        # Coordination Strategy: Avoid scheduling in the same slot as neighbors if they have specific preferences,
        # or try to balance load if neighbors are aiming for highly contrasted slots.
        
        # Neighbors' inferred priorities (based on ground truth):
        # N3: Prefers [1, 3]. Ground truth favors [0, 1, 2] depending on the day. Seems flexible but avoids the most expensive slots.
        # N5: Strongly prefers [0, 1]. Ground truth heavily favors slot 0.
        
        coord_cost = 0
        
        # Strategy: If a neighbor strongly prefers a slot, we penalize using that slot unless our comfort score is very high there.
        # Since we are in position 4, we prioritize our comfort (Slot 0, 3 high demand).
        
        # N5 strongly prefers slot 0.
        if slot_index == 0 and N5_PREF_SLOT_0: # N5 likely uses Slot 0 often
             coord_cost += self.gamma * 0.5 # Mild penalty for collision on N5's favorite
             
        # N3 prefers slot 1.
        if slot_index == 1 and N3_PREF_SLOT_1: # N3 likely uses Slot 1 often
             coord_cost += self.gamma * 0.3 # Slight penalty

        # Alternative Coordination (Load Balancing proxy):
        # Check spatial carbon map to see if the local area (Location 4) has high carbon usage.
        # Spatial carbon key format: '1: c1, c2, c3, c4; 2: ...; 4: ...'
        spatial_carbon_str = day_data['Spatial carbon'].get(str(self.scenario['location']))
        if spatial_carbon_str:
            spatial_data = [int(c) for c in spatial_carbon_str.split(';')[slot_index].split(',')]
            avg_spatial_carbon = sum(spatial_data) / len(spatial_data)
            # Penalize using this slot if local carbon is already high (assuming neighbors are also high load)
            if avg_spatial_carbon > 550:
                coord_cost += self.gamma * 0.1 * (avg_spatial_carbon / 750)


        # Total Cost Function: J = alpha * C_comfort + C_carbon + beta * C_congestion + gamma * C_coord
        total_cost = comfort_cost + carbon_cost + congestion_cost + coord_cost
        
        return total_cost

    def recommend_slots(self):
        recommendations = []
        
        # Pre-calculate neighbor preferences based on observed history (a very simple heuristic)
        # N3 Ground Truth favors: Slot 2 (Day 1), Slot 0 (Day 2), Slot 1 (Day 3), Slot 3 (Day 4), Slot 0 (Day 5), Slot 1 (Day 6), Slot 2 (Day 7)
        # N5 Ground Truth favors: Slot 0 (Days 1-5), Slot 1 (Days 6-7)
        
        # For simplicity in Stage 3 (where coordination is assumed minimal look-ahead):
        # N3 seems to use slots 0, 1, 2 somewhat evenly, often avoiding the absolute highest carbon/price (Slot 3).
        N3_PREF_SLOT_1 = True # Assume N3 uses slot 1 often based on stated preference and history spread.
        # N5 heavily favors slot 0.
        N5_PREF_SLOT_0 = True 

        for day_idx, day_key in enumerate(self.day_keys):
            day_data = self.scenario['days'][day_key]
            
            # Calculate costs for all slots for the current day
            slot_costs = {}
            for slot in range(self.num_slots):
                cost = self._calculate_cost(day_idx, slot, day_data)
                slot_costs[slot] = cost

            # Sort slots by calculated cost (minimize cost)
            sorted_slots = sorted(slot_costs.items(), key=lambda item: item[1])
            
            # Select the best slot index
            best_slot = sorted_slots[0][0]
            recommendations.append(best_slot)

        return recommendations

    def save_output(self, recommendations):
        output = {
            "agent_id": "Agent 4",
            "location": self.scenario.get('location'),
            "scenario_id": self.scenario.get('scenario_id'),
            "recommendations": [
                {"day": day_key, "slot_index": rec}
                for day_key, rec in zip(self.day_keys, recommendations)
            ]
        }
        
        # Save the slot indices only for the required JSON format
        # Required format: list of seven slot indices
        final_slot_list = recommendations
        
        with open("global_policy_output.json", 'w') as f:
            json.dump(final_slot_list, f, indent=4)

if __name__ == "__main__":
    policy = EVPolicy()
    if policy.scenario:
        slot_recommendations = policy.recommend_slots()
        policy.save_output(slot_recommendations)
# The final output only requires the policy.py script content.
# Based on the weights (alpha=40 high, gamma=12 moderate), comfort (high base load in 0 & 3) 
# and carbon intensity (grid objective) will heavily influence the decision, with slight coordination nudges.

# Expected behavior based on cost function:
# Persona prefers Slot 0 and 3 (Base load highest). Comfort cost is lowest here.
# Carbon/Price: Generally favors slots with lower values (e.g., Slot 1 often has lower carbon/price than Slot 3).

# Day 1: Prices [0.20, 0.25, 0.29, 0.32], Carbon [490, 470, 495, 540]
# Slot 0: Low Carbon, Moderate Comfort Cost
# Slot 3: High Carbon, Low Comfort Cost
# Likely Slot 0 or 1 due to low carbon/price, slightly favoring Slot 0 due to high comfort weight.

# Day 4: Prices [0.19, 0.24, 0.28, 0.22], Carbon [495, 470, 500, 535]
# Slot 3 has lowest price (0.22), Slot 1 has lowest carbon (470).
# High comfort weight on Slot 0 (0.9) and Slot 3 (0.8).
# Expected to pick Slot 0 or Slot 3, likely 0 due to slightly better carbon profile than 3.
# Coordination nudges might slightly push away from N5's slot 0 if N5 is known to be highly aggressive on 0.

# Since N5 strongly prefers 0, and N4 values comfort highly in 0, there might be a push/pull.
# Given the high comfort weight (alpha=40) relative to coordination (gamma=12), Agent 4 will strongly favor slots 0 and 3, unless carbon/price is extremely high in those slots.

# Running the logic internally based on cost:
# Day 1: Cost(0) < Cost(1) < Cost(2) < Cost(3) (Favors 0)
# Day 2: Cost(0) < Cost(1) < Cost(2) < Cost(3) (Favors 0)
# Day 3: Cost(1) < Cost(0) < Cost(2) < Cost(3) (Favors 1 due to low carbon in Slot 1=455 vs Carbon(0)=500)
# Day 4: Cost(0) < Cost(3) < Cost(1) < Cost(2) (Favors 0)
# Day 5: Cost(1) < Cost(0) < Cost(2) < Cost(3) (Favors 1 due to low carbon in Slot 1=450 vs Carbon(0)=500)
# Day 6: Cost(1) < Cost(0) < Cost(2) < Cost(3) (Favors 1)
# Day 7: Cost(1) < Cost(0) < Cost(3) < Cost(2) (Favors 1)

# Predicted Recommendation: [0, 0, 1, 0, 1, 1, 1]
# This heavily favors the agent's comfort slots (0 and 3) but pulls to slot 1 when carbon/price is significantly better there.
# Since Slot 3 is expensive/high carbon every day, Slot 0 is preferred over 3 for comfort. Slot 1 is often the best carbon compromise.
# Coordination attempts to slightly reduce use of Slot 0 (N5's favorite) and Slot 1 (N3's favorite).
# Since Slot 0 has the highest comfort weight, it should win over Slot 1 unless carbon difference is large.

# Re-evaluating the coordination effect:
# If N5 is in 0 often, N4 might pull away from 0. If N3 is in 1 often, N4 might pull away from 1.
# If coordination pulls N4 away from 0 and 1, the decision leans towards 3 (high comfort, high carbon) or 2 (low comfort, middle carbon).
# Given alpha=40, 3 should still be favored over 2 unless coordination cost in 3 is massive (it isn't).

# Final internal heuristic result based on cost structure favoring comfort highly: [0, 0, 0, 0, 0, 0, 3] (Strong comfort bias)
# Let's stick to the calculated cost minimizing strategy: [0, 0, 1, 0, 1, 1, 1]
# We will output the result of the simulation run by the logic defined in _calculate_cost.

import json
import os

class EVPolicy:
    def __init__(self):
        # Parameters from the prompt
        self.alpha = 40.00  # Comfort/Carbon weight (High weight for comfort/local factors)
        self.beta = 0.50    # Congestion weight (Low weight)
        self.gamma = 12.00  # Neighbor coordination weight

        self.num_slots = 4
        self.capacity = 6.8
        self.base_demand = [0.90, 0.60, 0.70, 0.80] # High preference for 0 and 3

        self.neighbor_examples = [
            {
                "name": "Neighbor 3",
                "location": 3,
                "base_demand": [0.60, 0.80, 0.90, 0.70],
                "preferred_slots": [1, 3],
                "comfort_penalty": 0.20,
                "ground_truth_min_cost_slots_by_day": [2, 0, 1, 3, 0, 1, 2]
            },
            {
                "name": "Neighbor 5",
                "location": 5,
                "base_demand": [0.50, 0.70, 0.60, 0.90],
                "preferred_slots": [0, 1],
                "comfort_penalty": 0.12,
                "ground_truth_min_cost_slots_by_day": [0, 0, 0, 0, 0, 1, 1]
            }
        ]

        self.scenario = self._load_scenario()
        self.day_keys = list(self.scenario['days'].keys())
        
        # Simple coordination assumptions based on neighbors' known historical leanings
        self.N3_PREF_SLOT_1 = True 
        self.N5_PREF_SLOT_0 = True 

    def _load_scenario(self):
        try:
            with open("scenario.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # In a real execution environment, this should not happen if the file structure is correct.
            return None
        except json.JSONDecodeError:
            return None

    def _calculate_cost(self, slot_index, day_data):
        # --- 1. Comfort Cost (Local Goal: High priority for Agent 4 based on alpha=40) ---
        comfort_preference_score = self.base_demand[slot_index]
        max_pref = max(self.base_demand)
        # Lower score means higher comfort cost (deviation from preferred slots 0, 3)
        comfort_cost = self.alpha * (max_pref - comfort_preference_score)

        # --- 2. Carbon Cost (Grid Goal) ---
        carbon_intensity = day_data['Carbon'][slot_index]
        carbon_cost = carbon_intensity 

        # --- 3. Congestion Cost (Grid Goal) ---
        agent_load_proxy = self.base_demand[slot_index]
        baseline = day_data['Baseline load'][slot_index]
        total_load_proxy = baseline + agent_load_proxy
        congestion_cost = self.beta * (total_load_proxy / self.capacity)

        # --- 4. Neighbor Coordination Cost (Collective Goal) ---
        coord_cost = 0
        
        # Penalize overlapping use of slots heavily favored by neighbors
        if slot_index == 0 and self.N5_PREF_SLOT_0: # N5 strongly favors slot 0
             coord_cost += self.gamma * 0.7 # Strong penalty for collision on N5's favorite slot
             
        if slot_index == 1 and self.N3_PREF_SLOT_1: # N3 prefers slot 1
             coord_cost += self.gamma * 0.4 # Moderate penalty

        # Spatial Carbon check (Loc 4) - Encourages shifting if local area is stressed
        spatial_carbon_str = day_data['Spatial carbon'].get(str(self.scenario['location']))
        if spatial_carbon_str:
            # We assume neighbors are distributed across other locations 1, 2, 3, 5
            # We look at the carbon associated with our location (index 4 in the spatial data list, which corresponds to position 4)
            # The format is '1: c1, c2, c3, c4; 2: ...; 4: c1, c2, c3, c4; ...'
            
            # Find the data block for Location 4
            loc_data_block = next((block for block in spatial_carbon_str.split(';') if block.startswith('4:')), None)
            
            if loc_data_block:
                # Extract the 4 values associated with Location 4's load profile across slots
                try:
                    # The slot data is split by comma, we want the value corresponding to the current slot index
                    spatial_values = [int(c.strip()) for c in loc_data_block.split(':')[1].split(',')]
                    if slot_index < len(spatial_values):
                        local_spatial_carbon = spatial_values[slot_index]
                        # If local carbon intensity is high, add a small coordination penalty (we are leading by example)
                        if local_spatial_carbon > 550:
                            coord_cost += self.gamma * 0.1 * (local_spatial_carbon / 750)
                except Exception:
                    pass # Ignore parsing errors if data format is unexpected

        # Total Cost Function: J = alpha*C_comfort + C_carbon + beta*C_congestion + gamma*C_coord
        total_cost = comfort_cost + carbon_cost + congestion_cost + coord_cost
        
        return total_cost

    def recommend_slots(self):
        recommendations = []
        
        for day_idx, day_key in enumerate(self.day_keys):
            day_data = self.scenario['days'][day_key]
            
            slot_costs = {}
            for slot in range(self.num_slots):
                cost = self._calculate_cost(slot, day_data)
                slot_costs[slot] = cost

            # Select the slot with the minimum total cost
            best_slot = min(slot_costs, key=slot_costs.get)
            recommendations.append(best_slot)

        return recommendations

    def save_output(self, recommendations):
        # Save the list of seven slot indices as required by the output specification
        with open("global_policy_output.json", 'w') as f:
            json.dump(recommendations, f, indent=4)

if __name__ == "__main__":
    policy = EVPolicy()
    if policy.scenario:
        slot_recommendations = policy.recommend_slots()
        policy.save_output(slot_recommendations)