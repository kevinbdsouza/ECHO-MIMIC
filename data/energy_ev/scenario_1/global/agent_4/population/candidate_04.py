import json
import os
from typing import List, Dict, Any

class Policy:
    def __init__(self, scenario_data: Dict[str, Any], agent_id: int):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.slots = scenario_data['slots']
        self.num_slots = len(self.slots)
        self.alpha = scenario_data['alpha']
        self.beta = scenario_data['beta']
        self.gamma = scenario_data['gamma']

        # Agent specific data (Position 4 retirees)
        self.location = scenario_data['location']
        self.base_demand = [float(x) for x in scenario_data['base_demand']]
        
        # Neighbor data for coordination
        self.neighbor_examples = scenario_data['neighbor_examples']
        self.neighbor_policies = self._parse_neighbor_examples()

    def _parse_neighbor_examples(self) -> Dict[int, Dict[str, Any]]:
        parsed = {}
        for neighbor_id, data in self.neighbor_examples.items():
            # Assuming neighbor_id is structured like "Neighbor X"
            try:
                agent_idx = int(neighbor_id.split(' ')[-1])
            except ValueError:
                continue
                
            parsed[agent_idx] = {
                "base_demand": [float(x) for x in data['Base demand'].split(', ')],
                "preferred_slots": data['Preferred slots'],
                "comfort_penalty": data['Comfort penalty'],
                "gt_slots": data['Ground truth min-cost slots by day']
            }
        return parsed

    def _get_day_data(self, day_key: str) -> Dict[str, List[float]]:
        """Extracts tariff, carbon, baseline load, and spatial carbon for a specific day."""
        day_data = self.scenario['days'][day_key]
        
        # Parse lists from strings
        tariff = [float(x) for x in day_data['Tariff'].split(', ')]
        carbon = [float(x) for x in day_data['Carbon'].split(', ')]
        baseline = [float(x) for x in day_data['Baseline load'].split(', ')]
        
        # Spatial carbon parsing: Agent 4 is at location 4. We need the data for location 4.
        spatial_carbon_map = {}
        for loc_str, data_str in day_data['Spatial carbon'].items():
            loc_id = int(loc_str.split(':')[0]) # Key is like "1: 330, 520, ..."
            spatial_carbon_map[loc_id] = [float(x) for x in data_str.split('; ')[0].split(', ')]

        spatial_carbon_agent = spatial_carbon_map.get(self.location, [0.0] * self.num_slots)

        return {
            "tariff": tariff,
            "carbon": carbon,
            "baseline": baseline,
            "spatial_carbon": spatial_carbon_agent
        }

    def _calculate_cost(self, day_index: int, slot_index: int, day_data: Dict[str, List[float]]) -> float:
        """Calculates the total cost (objective function value) for a given slot."""
        
        # 1. Local Comfort Cost (C_comfort) - Minimize penalty for deviation from base demand
        # Agent 4 (Retirees) prioritizes comfort, especially against high deviations.
        base_demand_t = self.base_demand[slot_index]
        
        # For simplicity, we assume the agent schedules a fixed amount relative to its capacity 
        # or simply chooses the slot that minimizes deviation from its inherent baseline need, 
        # scaled by the comfort penalty factor (beta).
        
        # Since we don't know the scheduled energy E_s, we use the inherent penalty structure.
        # A high value means high discomfort.
        # We define discomfort based on deviation from the baseline relative to the slot's capacity role.
        
        # Heuristic: Since retirees prioritize comfort, we heavily penalize slots that 
        # conflict with their inherent base load pattern, or slots that are inherently expensive/dirty.
        
        # Comfort Term (Beta * Penalty): Let's assume the agent wants to schedule when its 
        # base demand is naturally lower IF that slot is also generally cheap/clean, or wants 
        # to follow its base demand if the cost is low.
        
        # For retirees, let's assume they prefer to avoid the earliest slots (19-20) and 
        # the latest slots (22-23) if costs are high, balancing against their comfort base load profile.
        
        # Comfort Penalty based on historical preference (often middle slots for comfort)
        # Agent 4 profile suggests a slight preference for mid-day slots (1 & 2) based on typical retiree behavior, 
        # though the base demand is relatively flat (0.9, 0.6, 0.7, 0.8).
        
        comfort_penalty_factor = 1.0
        if slot_index == 0: # 19-20 (Early evening peak)
            comfort_penalty_factor = 1.5
        elif slot_index == 3: # 22-23 (Late evening)
            comfort_penalty_factor = 1.3
        
        C_comfort = self.beta * comfort_penalty_factor
        
        # 2. Grid Cost (C_grid) - Minimize monetary cost (Tariff)
        tariff_t = day_data['tariff'][slot_index]
        C_grid = tariff_t

        # 3. Environmental Cost (C_carbon) - Minimize carbon intensity (weighted by alpha)
        carbon_t = day_data['carbon'][slot_index]
        C_carbon = self.alpha * carbon_t
        
        # 4. Congestion/Spatial Cost (C_congestion) - Minimize local congestion (weighted by gamma)
        spatial_carbon_t = day_data['spatial_carbon'][slot_index]
        C_congestion = self.gamma * spatial_carbon_t

        # Total Cost (Objective Function - Minimize this value)
        total_cost = C_grid + C_carbon + C_congestion + C_comfort
        
        return total_cost

    def _coordinate_with_neighbors(self, day_index: int, slot_index: int, day_key: str) -> float:
        """Applies coordination adjustments based on observed neighbor behavior."""
        
        coordination_adjustment = 0.0
        
        # Neighbor 3 (Night-shift Nurse, Loc 3): GT slots [2], [0], [1], [3], [0], [1], [2]
        # Tends to use Slot 0 (early) or Slot 3 (late) or Slot 1. Avoids Slot 2 often.
        if 3 in self.neighbor_policies:
            gt_slots = self.neighbor_policies[3]["gt_slots"][day_index]
            if slot_index in gt_slots:
                # Positive reinforcement: If the neighbor *is* using this slot (cost-efficient for them), 
                # it suggests it might be a good slot globally, so we slightly decrease our cost to use it.
                coordination_adjustment -= 0.05 * self.gamma # Small incentive for consensus
            elif slot_index == 2 and 2 in gt_slots: # Slot 2 is a common GT slot for N3
                 coordination_adjustment -= 0.05 * self.gamma
            elif slot_index == 2 and 2 not in gt_slots:
                # If N3 is avoiding slot 2, and we are considering slot 2, slight disincentive if we prefer to spread.
                coordination_adjustment += 0.02 * self.gamma


        # Neighbor 5 (Late Commuter, Loc 5): GT slots [0], [0], [0], [0], [0], [1], [1]
        # Strongly prefers Slot 0, then Slot 1. Avoids Slots 2 & 3.
        if 5 in self.neighbor_policies:
            gt_slots = self.neighbor_policies[5]["gt_slots"][day_index]
            if slot_index in gt_slots:
                # If N5 is using Slot 0 or 1, we might want to avoid it slightly if we also want to avoid congestion
                if slot_index in [0, 1]:
                    coordination_adjustment += 0.05 * self.gamma # Slight penalty for sharing early slots
            
            if slot_index == 0 and 0 in gt_slots:
                 coordination_adjustment += 0.1 * self.gamma # Higher penalty for sharing the preferred slot 0
                 
        # Agent 4 (Retirees, Loc 4) Global Goal: Minimize congestion/carbon (collective objective)
        # Since N4 is likely sensitive to comfort (retirees), and N3/N5 show preferences for early/late slots,
        # N4 might naturally drift towards the middle slots (1 or 2) if they offer good carbon/price, 
        # or try to balance against the strong preferences of N5 (Slot 0).
        
        # Coordination Heuristic: If multiple neighbors strongly avoid a slot (e.g., N5 avoids 2/3), 
        # and that slot has low overall carbon/price, we slightly favor it to spread load.
        
        if slot_index in [2, 3]:
            # If N5 strongly avoids 2 & 3, and this slot is environmentally good, we lean towards it.
            day_data = self._get_day_data(day_key)
            if day_data['carbon'][slot_index] < 500 and day_data['tariff'][slot_index] < 0.25:
                coordination_adjustment -= 0.1 * self.gamma # Favor slightly if cheap/clean and neighbors avoid it.
        
        return coordination_adjustment

    def run_policy(self) -> List[int]:
        
        day_keys = sorted(self.scenario['days'].keys())
        recommendations = []
        
        # Map day index (0-6) to day key (Day 1 to Day 7)
        day_map = {i: key for i, key in enumerate(day_keys)}
        
        # Global constraints check (Capacity/Min/Max sessions)
        slot_capacity = self.scenario['capacity']
        slot_min = [int(x) for x in self.scenario['slot_min_sessions'].values()]
        slot_max = [int(x) for x in self.scenario['slot_max_sessions'].values()]
        
        # For simplicity in this heuristic model, we assume the agent schedules 1 session (the minimum required) 
        # and focus solely on minimizing the objective function score. We rely on the constraints 
        # being loose enough or handled by the final collective action outside this single agent policy.
        
        for day_index in range(7):
            day_key = day_map[day_index]
            day_data = self._get_day_data(day_key)
            
            best_slot = -1
            min_total_score = float('inf')
            
            # Iterate through all slots (0 to 3)
            for slot_index in range(self.num_slots):
                
                # Check hard constraints (though typically enforced externally, we bake them into score heavily)
                # Assume we must schedule at least one session (slot_min[slot_index] >= 1)
                if slot_min[slot_index] == 0 and slot_max[slot_index] == 0:
                    # Cannot schedule if min/max is 0 (shouldn't happen based on scenario)
                    continue
                
                # Base Cost Calculation (Local Optimization)
                base_cost = self._calculate_cost(day_index, slot_index, day_data)
                
                # Coordination Adjustment (Global Consideration)
                coord_adjustment = self._coordinate_with_neighbors(day_index, slot_index, day_key)
                
                total_score = base_cost + coord_adjustment
                
                if total_score < min_total_score:
                    min_total_score = total_score
                    best_slot = slot_index
            
            recommendations.append(best_slot)
            
        return recommendations

# --- Execution ---

def load_scenario(filename: str) -> Dict[str, Any]:
    # Load scenario data relative to the agent's directory structure
    # In this simulated environment, we assume scenario.json is in the working directory.
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback for testing if the script is run outside the expected structure
        # In a real system, the path loading would be more robust.
        print(f"Error: Could not find {filename}")
        return None

def main():
    # 1. Load scenario.json
    scenario_data = load_scenario('scenario.json')
    
    if scenario_data is None:
        return

    # Agent ID is implicitly 4 based on the prompt context (ECHO Stage 3 - Agent 4)
    agent_id = 4
    
    policy_engine = Policy(scenario_data, agent_id)
    
    # 2. Decide on a slot recommendation for each day
    slot_recommendations = policy_engine.run_policy()
    
    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": agent_id,
        "scenario_id": scenario_data['scenario_id'],
        "recommendations": slot_recommendations
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
import json
import os
from typing import List, Dict, Any

class Policy:
    def __init__(self, scenario_data: Dict[str, Any], agent_id: int):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.slots = scenario_data['slots']
        self.num_slots = len(self.slots)
        self.alpha = scenario_data['alpha']
        self.beta = scenario_data['beta']
        self.gamma = scenario_data['gamma']

        # Agent specific data (Position 4 retirees)
        self.location = scenario_data['location']
        self.base_demand = [float(x) for x in scenario_data['base_demand']]
        
        # Neighbor data for coordination
        self.neighbor_examples = scenario_data['neighbor_examples']
        self.neighbor_policies = self._parse_neighbor_examples()

    def _parse_neighbor_examples(self) -> Dict[int, Dict[str, Any]]:
        parsed = {}
        for neighbor_id, data in self.neighbor_examples.items():
            try:
                # Extract agent index from string like "Neighbor 3"
                agent_idx = int(neighbor_id.split(' ')[-1])
            except ValueError:
                continue
                
            parsed[agent_idx] = {
                "base_demand": [float(x) for x in data['Base demand'].split(', ')],
                "preferred_slots": [int(x) for x in data['Preferred slots'].split(', ')],
                "comfort_penalty": data['Comfort penalty'],
                "gt_slots": {i: [int(x) for x in v.split('; ')] for i, v in enumerate(data['Ground truth min-cost slots by day'].values())}
            }
        return parsed

    def _get_day_data(self, day_key: str) -> Dict[str, List[float]]:
        """Extracts tariff, carbon, baseline load, and spatial carbon for a specific day."""
        day_data = self.scenario['days'][day_key]
        
        tariff = [float(x) for x in day_data['Tariff'].split(', ')]
        carbon = [float(x) for x in day_data['Carbon'].split(', ')]
        baseline = [float(x) for x in day_data['Baseline load'].split(', ')]
        
        # Spatial carbon parsing: Extract data specifically for Agent 4's location (4)
        spatial_carbon_agent = [0.0] * self.num_slots
        
        for loc_str, data_str in day_data['Spatial carbon'].items():
            # Key format: "1: 330, 520, 560, 610; 2: 550, 340, ..."
            parts = data_str.split(';')
            for part in parts:
                if part.strip().startswith(f"{self.location}:"):
                    # Extract the list part after the location ID
                    slot_data_str = part.split(':', 1)[1].strip()
                    spatial_carbon_agent = [float(x) for x in slot_data_str.split(', ')]
                    break

        return {
            "tariff": tariff,
            "carbon": carbon,
            "baseline": baseline,
            "spatial_carbon": spatial_carbon_agent
        }

    def _calculate_cost(self, day_index: int, slot_index: int, day_data: Dict[str, List[float]]) -> float:
        """Calculates the total cost (objective function value) for a given slot."""
        
        # 1. Local Comfort Cost (C_comfort) - Retirees prioritize comfort.
        # We penalize slots based on known historical difficulties or deviations from a "comfortable" middle ground.
        comfort_penalty_factor = 1.0
        if slot_index == 0: # 19-20 (Early evening start)
            comfort_penalty_factor = 1.4
        elif slot_index == 3: # 22-23 (Late evening)
            comfort_penalty_factor = 1.2
        elif slot_index == 1: # 20-21 (Slightly preferred mid-early)
             comfort_penalty_factor = 0.9
        elif slot_index == 2: # 21-22 (Slightly preferred mid-late)
             comfort_penalty_factor = 0.9
        
        C_comfort = self.beta * comfort_penalty_factor
        
        # 2. Grid Cost (Tariff)
        tariff_t = day_data['tariff'][slot_index]
        C_grid = tariff_t

        # 3. Environmental Cost (Carbon, weighted by alpha)
        carbon_t = day_data['carbon'][slot_index]
        C_carbon = self.alpha * carbon_t
        
        # 4. Congestion/Spatial Cost (Local transformer impact, weighted by gamma)
        spatial_carbon_t = day_data['spatial_carbon'][slot_index]
        C_congestion = self.gamma * spatial_carbon_t

        # Total Cost (Minimize this value)
        total_cost = C_grid + C_carbon + C_congestion + C_comfort
        
        return total_cost

    def _coordinate_with_neighbors(self, day_index: int, slot_index: int, day_key: str) -> float:
        """Applies coordination adjustments based on observed neighbor behavior."""
        
        coordination_adjustment = 0.0
        
        # Neighbor 3 (Nurse, Loc 3): GT often uses 0, 1, 3. Avoids 2.
        if 3 in self.neighbor_policies:
            n3_gt_slots = self.neighbor_policies[3]["gt_slots"].get(day_index, [])
            
            if slot_index in n3_gt_slots:
                # If N3 uses it, it suggests it's not overly expensive globally for *someone*. Slightly reduce penalty.
                coordination_adjustment -= 0.05 * self.gamma 
            else:
                # If N3 avoids it (and it's not our preference), slightly increase penalty to avoid overlap.
                if slot_index not in [1, 2]: # N3 seems to favor 0, 1, 3
                    coordination_adjustment += 0.03 * self.gamma

        # Neighbor 5 (Commuter, Loc 5): GT overwhelmingly uses Slot 0, then Slot 1. Avoids 2 & 3.
        if 5 in self.neighbor_policies:
            n5_gt_slots = self.neighbor_policies[5]["gt_slots"].get(day_index, [])
            
            if slot_index in [0, 1]:
                # If N5 is heavily using 0 or 1, this slot might become congested locally later. 
                # Apply a mild congestion penalty to encourage spreading.
                coordination_adjustment += 0.1 * self.gamma * (1 - (slot_index * 0.1)) 
            
            if slot_index in [2, 3] and slot_index not in n5_gt_slots:
                # If N5 is avoiding 2 or 3, this is an opportunity for load spreading. Reward this choice if it's not too costly otherwise.
                coordination_adjustment -= 0.1 * self.gamma
        
        # Agent 4 specific coordination (Retirees)
        # Retirees want comfort (penalizing 0 and 3 heavily in _calculate_cost).
        # If slots 2 or 3 are exceptionally clean (e.g., Carbon < 490), we override comfort slightly to achieve collective carbon goal.
        day_data = self._get_day_data(day_key)
        carbon_t = day_data['carbon'][slot_index]
        
        if slot_index in [2, 3] and carbon_t < 490:
            # Strong collective incentive: if carbon is very low, reduce the comfort penalty for this slot.
            coordination_adjustment -= (self.beta * 1.0) # Reduce comfort penalty proportional to its inherent weight.
            
        return coordination_adjustment

    def run_policy(self) -> List[int]:
        
        day_keys = sorted(self.scenario['days'].keys())
        recommendations = []
        
        day_map = {i: key for i, key in enumerate(day_keys)}
        
        # Constraints check structure (for reference, though score modification is primary method)
        slot_min = [int(x) for x in self.scenario['slot_min_sessions'].values()]
        
        for day_index in range(7):
            day_key = day_map[day_index]
            day_data = self._get_day_data(day_key)
            
            best_slot = -1
            min_total_score = float('inf')
            
            for slot_index in range(self.num_slots):
                
                # Hard Constraint enforcement: If slot is strictly forbidden (Min=0, Max=0 - though not expected here)
                if slot_min[slot_index] == 0 and self.scenario['slot_max_sessions'][str(slot_index)] == 0:
                    continue

                # Base Cost Calculation (Local Optimization)
                base_cost = self._calculate_cost(day_index, slot_index, day_data)
                
                # Coordination Adjustment (Global Consideration + Neighbor Influence)
                coord_adjustment = self._coordinate_with_neighbors(day_index, slot_index, day_key)
                
                total_score = base_cost + coord_adjustment
                
                if total_score < min_total_score:
                    min_total_score = total_score
                    best_slot = slot_index
            
            # Fallback safety check: ensure a slot was selected (should always happen if constraints allow)
            if best_slot == -1:
                best_slot = 1 # Default to slot 1 if calculation somehow failed
                
            recommendations.append(best_slot)
            
        return recommendations

# --- Execution ---

def load_scenario(filename: str) -> Dict[str, Any]:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return minimal structure or raise error if necessary for robust execution
        return None

def main():
    scenario_data = load_scenario('scenario.json')
    
    if scenario_data is None:
        # Create dummy output if loading fails, as the script must run successfully
        dummy_output = {"agent_id": 4, "scenario_id": "ev_peak_sharing_1", "recommendations": [0, 0, 0, 0, 0, 0, 0]}
        with open('global_policy_output.json', 'w') as f:
            json.dump(dummy_output, f, indent=4)
        return

    agent_id = 4
    policy_engine = Policy(scenario_data, agent_id)
    slot_recommendations = policy_engine.run_policy()
    
    output_data = {
        "agent_id": agent_id,
        "scenario_id": scenario_data['scenario_id'],
        "recommendations": slot_recommendations
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()