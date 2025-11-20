import json
import os
import numpy as np

class DERPolicy:
    def __init__(self, scenario_data, agent_id):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.T = len(scenario_data['slots'])
        
        # Agent specific data (Agent 3 Context)
        self.base_demand = np.array(self.scenario['profile']['base_demand'])
        self.capacity = self.scenario['capacity']
        self.alpha = self.scenario['alpha']   # Carbon weight
        self.beta = self.scenario['beta']     # Price weight
        self.gamma = self.scenario['gamma']   # Congestion/Comfort weight
        self.location = self.scenario['profile']['location'] # Location 3

        # Fixed neighbor data structures
        self.neighbor_data = self._process_neighbor_examples(self.scenario.get('neighbor_examples', {}))
        self.day_keys = sorted([k for k in self.scenario['days'].keys() if k.startswith("Day")])

    def _process_neighbor_examples(self, neighbors):
        processed = {}
        # Neighbors provided are 2 and 5
        for name, data in neighbors.items():
            match = name.split(' — ')[0].split(' ')[-1] # Extracts '2' or '5'
            processed[int(match)] = {
                'base_demand': np.array(data['Base demand']),
                'preferred_slots': set(data['Preferred slots']),
                'ground_truth': data['Ground truth min-cost slots by day']
            }
        return processed

    def _get_day_data(self, day_index):
        day_key = self.day_keys[day_index]
        day_data = self.scenario['days'][day_key]
        
        tariffs = np.array(day_data['Tariff'])
        carbons = np.array(day_data['Carbon'])
        baselines = np.array(day_data['Baseline load'])
        
        spatial_carbon = {}
        spat_str = day_data['Spatial carbon']
        
        # Parse spatial carbon: "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ..."
        for item in spat_str.split(';'):
            if ':' in item:
                loc_id, c_str = item.strip().split(':')
                spatial_carbon[int(loc_id)] = np.array([float(c.strip()) for c in c_str.split(',')])
            
        return tariffs, carbons, baselines, spatial_carbon

    def _get_neighbor_target_slot(self, day_idx, neighbor_id):
        """Infers the slot chosen by a specific neighbor based on their ground truth."""
        if neighbor_id in self.neighbor_data:
            gt_slots = self.neighbor_data[neighbor_id]['ground_truth']
            if day_idx < len(gt_slots):
                return gt_slots[day_idx]
        return -1

    def _calculate_agent_cost(self, day_idx, slot_idx, spatial_carbon, neighbor_load_at_slot):
        
        tariffs, carbons, _, _ = self._get_day_data(day_idx)
        
        price = tariffs[slot_idx]
        carbon = carbons[slot_idx]
        
        # Agent Demand Contribution (1.0 if charging, 0.0 otherwise)
        demand_contribution = self.base_demand[slot_idx]
        
        # --- 1. Global Objective Costs (Carbon & Price) ---
        # Use Carbon intensity weighted by Alpha, Price weighted by Beta
        individual_cost = (self.alpha * carbon) + (self.beta * price)
        
        # --- 2. Local Congestion Cost (Spatial Carbon & Feeder Load) ---
        
        # Spatial Carbon Cost (Weighted by Gamma, related to local transformer stress at location 3)
        loc_carbon = spatial_carbon.get(self.location, self.scenario['carbon_intensity'])[slot_idx]
        spatial_cost = self.gamma * (loc_carbon / 1000.0) 

        # Feeder Congestion Cost (Based on neighbor activity + self)
        total_concurrent_load = demand_contribution + neighbor_load_at_slot
        
        # Use capacity to normalize congestion. Penalize heavily if exceeding 100% of capacity.
        congestion_ratio = total_concurrent_load / self.capacity
        congestion_penalty = self.gamma * max(0, congestion_ratio - 1.0) * 5.0 # High penalty factor (5.0) for exceeding capacity

        # --- 3. Comfort/Preference Cost (Agent 3: Nurse, Location 3) ---
        # Base demand profile: [0.60, 0.80, 0.90 (S2), 0.70]. Slot 2 is peak demand/comfort.
        comfort_penalty = 0.0
        if slot_idx != 2:
            # Penalize deviation from peak demand slot (Slot 2) by Gamma weight
            comfort_penalty = self.gamma * abs(slot_idx - 2) * 1.5 
        
        total_cost = individual_cost + spatial_cost + congestion_penalty + comfort_penalty
        
        return total_cost

    def run_day_optimization(self, day_idx):
        tariffs, carbons, baselines, spatial_carbon = self._get_day_data(day_idx)
        
        best_slot = -1
        min_cost = float('inf')
        
        # --- Coordination Strategy: Avoid conflicts with known neighbor GT slots if global objectives are similar ---
        
        # Calculate the cost for every slot, including coordination penalties based on neighbor GT decisions
        for s in range(self.T):
            
            # 1. Estimate Neighbor Load if Agent 3 chooses slot 's'
            neighbor_load_at_slot = 0.0
            for neighbor_id in self.neighbor_data.keys():
                target_slot = self._get_neighbor_target_slot(day_idx, neighbor_id)
                
                if target_slot == s:
                    # Neighbor is charging in this slot. Use their base demand as the load estimate.
                    neighbor_load_at_slot += self.neighbor_data[neighbor_id]['base_demand'][s]
            
            # 2. Calculate Base Cost
            base_cost = self._calculate_agent_cost(day_idx, s, spatial_carbon, neighbor_load_at_slot)
            
            # 3. Coordination Adjustment
            coordination_adjustment = 0.0
            
            # Check for *clash* with neighbor preferred slots, especially if the slot is globally expensive.
            is_clash = False
            for neighbor_id, data in self.neighbor_data.items():
                gt_slot = self._get_neighbor_target_slot(day_idx, neighbor_id)
                if gt_slot == s:
                    # If we choose the slot the neighbor chose (based on GT), apply a small penalty 
                    # to encourage spreading out, unless our comfort score is very low (i.e., base cost is very high).
                    
                    # Only penalize if the slot is *not* significantly better for the agent (Slot 2 is preferred)
                    if s != 2:
                        coordination_adjustment += self.gamma * 0.5 * data['comfort_penalty']
                    else:
                        # If we clash on our preferred slot (S2), we assume this is necessary, 
                        # and only penalize slightly based on neighbor's *own* perceived penalty for choosing it.
                        coordination_adjustment += self.gamma * 0.1 * data['comfort_penalty']


            final_cost = base_cost + coordination_adjustment
            
            if final_cost < min_cost:
                min_cost = final_cost
                best_slot = s
                
        return best_slot

    def generate_recommendation(self):
        recommendations = []
        for day_idx in range(len(self.day_keys)):
            slot = self.run_day_optimization(day_idx)
            recommendations.append(slot)
        return recommendations

def load_scenario(file_path):
    """Loads scenario data, providing required structure for the policy class."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        # Minimal structure to allow class instantiation for testing if needed, but exiting is safer.
        exit(1)
    
    # Restructure to match expectations if necessary, or assume direct load works.
    # Based on Candidate 1, direct loading seems sufficient if keys exist.
    
    # Add agent context for Agent 3 (Location 3, Base Demand)
    data['profile'] = {
        "location": 3,
        "base_demand": [0.60, 0.80, 0.90, 0.70]
    }
    # Ensure alpha, beta, gamma are present if not already in the header structure
    if 'alpha' not in data:
        data['alpha'] = 40.00
        data['beta'] = 0.50
        data['gamma'] = 12.00
        
    return data

def main():
    scenario_file = 'scenario.json'
    AGENT_ID = 3 
    
    scenario_data = load_scenario(scenario_file)

    policy = DERPolicy(scenario_data, AGENT_ID)
    recommendations = policy.generate_recommendation()

    # Output specification: global_policy_output.json containing a list of seven slot indices
    # Note: The prompt asks for a list of seven slot indices, not a complex JSON structure.
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(recommendations, f, indent=4)

if __name__ == "__main__":
    main()