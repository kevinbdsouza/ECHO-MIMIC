import json
import os
import numpy as np

class DERPolicy:
    def __init__(self, scenario_data, agent_id):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.T = len(scenario_data['slots'])
        self.N_neighbors = len(scenario_data.get('neighbor_examples', {}))

        # Agent specific data
        self.base_demand = np.array(self.scenario['base_demand'])
        self.capacity = self.scenario['capacity']
        self.alpha = self.scenario['alpha']
        self.beta = self.scenario['beta']
        self.gamma = self.scenario['gamma']

        # Fixed neighbor data (for this agent configuration)
        self.neighbor_data = self._process_neighbor_examples(self.scenario.get('neighbor_examples', {}))

    def _process_neighbor_examples(self, neighbors):
        processed = {}
        for name, data in neighbors.items():
            match = name.split(' — ')[0].split(' ')[-1]
            processed[match] = {
                'base_demand': np.array(data['Base demand']),
                'preferred_slots': set(data['Preferred slots']),
                'comfort_penalty': data['Comfort penalty'],
                'ground_truth': data['Ground truth min-cost slots by day']
            }
        return processed

    def _get_day_data(self, day_index):
        day_key = f"Day {day_index + 1}"
        
        # Find the key that matches the day description format
        matching_day_key = next((k for k in self.scenario['days'] if k.startswith(day_key)), None)
        if not matching_day_key:
            raise ValueError(f"Could not find data for {day_key}")
            
        day_data = self.scenario['days'][matching_day_key]
        
        tariffs = np.array(day_data['Tariff'])
        carbons = np.array(day_data['Carbon'])
        baselines = np.array(day_data['Baseline load'])
        
        spatial_carbon = {}
        spat_str = day_data['Spatial carbon']
        
        # Parse spatial carbon: "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ..."
        for item in spat_str.split(';'):
            loc_id, c_str = item.strip().split(':')
            spatial_carbon[int(loc_id)] = np.array([int(c.strip()) for c in c_str.split(',')])
            
        return tariffs, carbons, baselines, spatial_carbon

    def _calculate_agent_cost(self, day_idx, slot_idx, baseline_load, spatial_carbon, neighbor_activity=None):
        
        # 1. Individual Cost Component (Comfort/Demand Satisfaction)
        # Assume the agent wants to use their base demand in the least costly slot available
        # Cost = alpha * Price + beta * Carbon + gamma * Congestion_Penalty
        
        # Price and Carbon from the scenario (assuming these are the *expected* future values)
        # Note: The scenario description suggests the main day data has specific values, 
        # but the "Forecast note" implies variance. We use the day-specific values if available, 
        # otherwise fall back to the header values for the common slots.
        
        # Use the specific day's data if available
        tariffs, carbons, _, _ = self._get_day_data(day_idx)
        
        price = tariffs[slot_idx]
        carbon = carbons[slot_idx]
        
        # Agent Demand (1.0 if charging, 0.0 otherwise)
        # Since we only recommend one slot, the demand contribution is simply the base demand if selected.
        demand_if_chosen = self.base_demand[slot_idx]
        
        # Comfort/Cost Penalty (Relative to baseline demand in that slot)
        # For simplicity in this collective model, we focus on the core objectives: Price, Carbon, Congestion.
        # Comfort is implicitly modeled by choosing slots that minimize overall stress/cost.
        
        # Individual Cost Calculation based on objectives:
        individual_cost = (self.alpha * price) + (self.beta * carbon)
        
        # 2. Congestion Cost Component (Based on Neighbor Activity & Capacity)
        congestion_penalty = 0.0
        if neighbor_activity is not None:
            # Calculate total concurrent load (including self if chosen)
            total_load = demand_if_chosen
            for neighbor_slot_idx, neighbor_demand in neighbor_activity.items():
                if neighbor_slot_idx == slot_idx:
                    total_load += neighbor_demand 
            
            # Congestion is related to how close we are to capacity, weighted by the gamma factor
            if self.capacity > 0:
                congestion_ratio = total_load / self.capacity
                # Apply penalty if load exceeds baseline or capacity (using ratio as proxy)
                congestion_penalty = self.gamma * max(0, congestion_ratio - 1.0)
        
        # 3. Spatial Congestion/Carbon Penalty (Specific to Agent Location 3)
        # Agent is at Location 3
        spatial_cost = 0.0
        if spatial_carbon and self.scenario['location'] in spatial_carbon:
            # Spatial carbon at slot_idx for this agent's location
            loc_carbon = spatial_carbon[self.scenario['location']][slot_idx]
            # Use a fraction of gamma for spatial impact, potentially using the grid carbon as a reference
            # Here, we prioritize high spatial carbon as undesirable
            spatial_cost = (self.gamma / 2.0) * (loc_carbon / 1000.0) # Normalize scale

        
        total_cost = individual_cost + congestion_penalty + spatial_cost
        
        # Add a penalty if the demand is incompatible with local constraints (e.g., too low/high demand for slot)
        # Given we must choose one slot, we don't implement a specific comfort term unless it's required to break ties.
        
        return total_cost

    def _get_neighbor_demand(self, day_idx, slot_idx):
        # Estimate neighbor demand based on their preferred slots and ground truth
        neighbor_loads = {}
        
        day_key_map = {
            0: 'Day 1', 1: 'Day 2', 2: 'Day 3', 3: 'Day 4', 
            4: 'Day 5', 5: 'Day 6', 6: 'Day 7'
        }
        day_label = day_key_map[day_idx]
        
        for name, data in self.neighbor_data.items():
            # Strategy: Assume neighbors charge their full base demand in their chosen slot.
            # Use ground truth if available to infer their likely action for coordination.
            
            gt_slots = data['ground_truth']
            
            # Determine the target slot for this neighbor today
            target_slot = -1
            try:
                # Find the index corresponding to the current day label
                day_idx_in_gt = [d for d in self.scenario['days'].keys() if d.startswith(day_label)][0]
                
                # The ground truth list is ordered Day 1 to Day 7
                gt_index = int(day_label.split(' ')[1]) - 1
                if gt_index < len(gt_slots):
                    target_slot = gt_slots[gt_index]
            except Exception:
                # Fallback: If GT indexing fails, use preferred slots or nearest slot
                if slot_idx in data['preferred_slots']:
                    target_slot = slot_idx
                else:
                    # If not in preferred, choose the *best* preferred slot for coordination
                    # Since we are minimizing cost, we assume neighbors pick their best cost slot
                    # For simplicity in prediction, let's assume they follow the GT exactly.
                    target_slot = gt_slots[day_idx] if day_idx < len(gt_slots) else list(data['preferred_slots'])[0]

            
            if target_slot == slot_idx:
                # Assume neighbor charges their full base demand in the chosen slot
                neighbor_loads[name] = data['base_demand'][slot_idx]
            else:
                # Assume neighbor charges 0 load if not in the calculated slot
                neighbor_loads[name] = 0.0
                
        return neighbor_loads

    def run_day_optimization(self, day_idx):
        tariffs, carbons, baselines, spatial_carbon = self._get_day_data(day_idx)
        
        best_slot = -1
        min_cost = float('inf')
        
        # --- 1. Calculate Neighbor Activity Baseline (Coordination Input) ---
        # We need to estimate what neighbors *will* do if we make a decision.
        # Since this is a collective stage, we assume neighbors coordinate based on their GT/preferences.
        
        # For simplicity, we run an iterative coordination guess, but since we only output *our* slot,
        # we calculate the neighbor load assuming they followed their *observed* best strategy (GT).
        neighbor_activity_baseline = self._get_neighbor_demand(day_idx, -1) # -1 means calculate for all slots

        # --- 2. Evaluate Each Slot for Agent 3 ---
        
        # Local Comfort/Demand Factor: Agent 3 (Nurse) prefers night shifts, low demand early morning slots (1, 2) 
        # Base demand: 0.60 (S0), 0.80 (S1), 0.90 (S2), 0.70 (S3)
        # Location 3: Prefers slots where spatial carbon is low (Day 3: S2 is low spatial)
        
        # Heuristic Adjustment: Prioritize slots 1 and 2 (high base demand implies higher need/comfort weight)
        # and slots with lower carbon intensity, while respecting capacity.
        
        for s in range(self.T):
            
            # Calculate neighbor load *if* Agent 3 selects slot s
            current_neighbor_loads = self._get_neighbor_demand(day_idx, s)
            
            # Cost evaluation (Includes carbon, price, and congestion based on predicted neighbors)
            cost = self._calculate_agent_cost(day_idx, s, baselines[s], spatial_carbon, current_neighbor_loads)
            
            # Coordination Heuristic: Favor slots that neighbors are *not* heavily using, 
            # especially if neighbors are observed to be prioritizing low-carbon slots on other days.
            
            # Coordination Check (Inferred Neighbor Load at Slot s)
            inferred_neighbor_load = sum(current_neighbor_loads.values())
            
            # Apply a slight bonus if the slot appears less congested based on neighbors' GT choices
            # (This encourages spreading out if neighbors are clustered, or clustering if neighbors are sparse)
            if inferred_neighbor_load > 0:
                 # If neighbors are present, penalize based on how full the slot is relative to capacity
                 coordination_bonus = -self.gamma * (inferred_neighbor_load / self.capacity) * 0.5
            else:
                 coordination_bonus = 0.0
                 
            final_cost = cost + coordination_bonus
            
            # Agent 3 specific weighting: Nurse working night shift (strong preference for S1/S2)
            # Introduce a strong penalty for S0 and S3 unless they offer massive savings.
            comfort_weight = 0.0
            if s == 0 or s == 3:
                comfort_weight = 50.0 # High penalty for undesirable slots
            elif s == 1 or s == 2:
                comfort_weight = -10.0 # Small reward for desired slots (S1/S2)
                
            final_cost += comfort_weight
            
            if final_cost < min_cost:
                min_cost = final_cost
                best_slot = s
                
        return best_slot

    def generate_recommendation(self):
        recommendations = []
        for day_idx in range(7):
            slot = self.run_day_optimization(day_idx)
            recommendations.append(slot)
        return recommendations

def load_scenario(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # The scenario file name is assumed to be 'scenario.json' relative to the execution directory
    scenario_file = 'scenario.json'
    
    # In a real environment, we would need the agent_id, but here we hardcode Agent 3 context
    AGENT_ID = 3 
    
    try:
        scenario_data = load_scenario(scenario_file)
    except FileNotFoundError:
        # Handle case where scenario.json might be expected in a different structure
        # For this specific problem, we assume it's in the current directory.
        print(f"Error: {scenario_file} not found.")
        return

    policy = DERPolicy(scenario_data, AGENT_ID)
    recommendations = policy.generate_recommendation()

    # Output specification: global_policy_output.json
    output_data = {
        "recommendations": recommendations
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()