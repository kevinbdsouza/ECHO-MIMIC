import json
import os
from typing import List, Dict, Any, Tuple

# --- Configuration ---
SLOT_COUNT = 4
CAPACITY = 6.8
SLOT_MIN_SESSIONS = [1, 1, 1, 1]
SLOT_MAX_SESSIONS = [2, 2, 1, 2]
ALPHA = 40.00
BETA = 0.50
GAMMA = 12.00

class EVPeakSharingAgent:
    def __init__(self, scenario_data: Dict[str, Any], agent_id: int):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.location_id = self.scenario['location']
        self.num_days = len(self.scenario['days'])
        self.slots_info = self._parse_slots()
        self.neighbor_examples = self._parse_neighbors()
        self.base_demand = self._get_agent_base_demand()

    def _parse_slots(self) -> Dict[str, List[float]]:
        return {
            'price': self.scenario['price'],
            'carbon': self.scenario['carbon_intensity'],
            'baseline_load': self.scenario['baseline_load'],
        }

    def _get_agent_base_demand(self) -> List[float]:
        # Agent 2 is at location 2
        return self.scenario['base_demand']

    def _parse_neighbors(self) -> List[Dict[str, Any]]:
        neighbors = []
        for name, data in self.scenario['neighbor_examples'].items():
            # Extract location ID from neighbor name format (e.g., "Position X")
            location_match = next((i + 1 for i, loc in enumerate(['1', '2', '3', '4', '5']) if loc in name), None)
            
            neighbors.append({
                'name': name,
                'location': location_match if location_match else None,
                'base_demand': data['Base demand'],
                'preferred_slots': data['Preferred slots'],
                'comfort_penalty': data['Comfort penalty'],
                'ground_truth': data['Ground truth min-cost slots by day']
            })
        return neighbors

    def _get_day_data(self, day_index: int, day_name: str) -> Dict[str, Any]:
        day_key = f"Day {day_index + 1} ({day_name.split('(')[0].strip()})"
        day_data = self.scenario['days'][day_key]
        
        # Parse spatial carbon data specific to this agent's location
        spatial_carbon_list = {}
        for s_id in range(1, 6):
            spatial_carbon_list[s_id] = [float(x) for x in day_data['Spatial carbon'][f'{s_id}:'][0].split(', ')]
            
        return {
            'tariff': day_data['Tariff'],
            'carbon': day_data['Carbon'],
            'baseline_load': day_data['Baseline load'],
            # Use the agent's specific spatial carbon value for the chosen location
            'spatial_carbon': spatial_carbon_list.get(self.location_id, self.scenario['carbon_intensity'])
        }

    def calculate_cost(self, day_index: int, slot_index: int, session_count: int) -> float:
        """
        Calculates the composite cost for a given slot and session count.
        
        Cost components:
        1. Local Cost (Price + Carbon): Weighted by alpha and beta.
        2. Congestion Cost (Baseline Load vs Capacity): Penalized if load exceeds capacity.
        3. Coordination Cost (Neighbor Deviation): Penalized based on neighbor behavior.
        """
        day_data = self._get_day_data(day_index, list(self.scenario['days'].keys())[day_index])
        
        # 1. Local Cost (Price and Carbon)
        price = day_data['tariff'][slot_index]
        carbon = day_data['carbon'][slot_index]
        
        # The agent prioritizes transformer headroom (congestion), suggesting a stronger focus on load/capacity.
        # We scale carbon relative to the base scenario carbon intensity for this slot.
        local_cost = (ALPHA * price) + (BETA * carbon)
        
        # 2. Congestion Cost (Transformer Headroom Focus - High Weight)
        # Agent 2 is a feeder analyst prioritizing transformer headroom (Capacity constraint)
        baseline = day_data['baseline_load'][slot_index]
        load_increase = session_count * self.base_demand[slot_index]
        total_load = baseline + load_increase
        
        # Severe penalty if capacity is exceeded, mild penalty otherwise (relative to capacity)
        congestion_penalty = 0.0
        if total_load > CAPACITY:
            # Heavy penalty for exceeding capacity
            congestion_penalty = 1000.0 * (total_load - CAPACITY)
        else:
            # Mild penalty for taking up headroom, scaled by how much space is left
            headroom_usage = total_load / CAPACITY
            congestion_penalty = GAMMA * headroom_usage 
            
        # 3. Coordination Cost (Spatial Carbon Awareness)
        # Penalize using slots where local spatial carbon is high relative to the global average for that slot.
        spatial_c = day_data['spatial_carbon'][slot_index]
        global_c = self.scenario['carbon_intensity'][slot_index]
        
        coord_cost = 0.0
        if spatial_c > global_c * 1.1: # If local carbon is significantly higher (10% threshold)
            coord_cost = 5.0 * (spatial_c - global_c)
            
        # 4. Neighbor Coordination (Implicitly handled by slot choice, but we add a minor penalty 
        # if we are moving drastically against neighbor consensus for a general minimum cost goal)
        # Since the primary goal is local headroom, we don't strongly penalize deviation from neighbors,
        # but we might slightly favor slots neighbors avoid if we have surplus capacity.
        
        total_cost = local_cost + congestion_penalty + coord_cost
        return total_cost

    def recommend_slot(self, day_index: int) -> int:
        """
        Chooses the best slot for the day by minimizing the composite cost, 
        respecting min/max session constraints (which we assume are handled externally 
        or implicitly by choosing 1 session if min_sessions=1).
        
        Since this is the initial policy decision, we assume 1 session unless 
        the agent needs to maximize load for congestion management. Given the 
        ECHO stage structure, we usually recommend the *best time slot* for a single session.
        We fix session_count = 1 for recommendation here, constrained by min/max sessions.
        """
        day_name = list(self.scenario['days'].keys())[day_index]
        
        # Determine session count for this recommendation (assuming minimum required for now)
        session_count = SLOT_MIN_SESSIONS[0] # Use the minimum required session count (1)

        # Check day-specific constraints (if available, although here only general are provided)
        min_s = SLOT_MIN_SESSIONS[slot_index]
        max_s = SLOT_MAX_SESSIONS[slot_index]
        
        best_slot = -1
        min_cost = float('inf')

        for slot in range(SLOT_COUNT):
            # Check constraints for the proposed session count (1)
            if session_count < SLOT_MIN_SESSIONS[slot] or session_count > SLOT_MAX_SESSIONS[slot]:
                continue

            cost = self.calculate_cost(day_index, slot, session_count)
            
            # Agent Specific Heuristic Consideration: Transformer Headroom (Congestion)
            # For Agent 2 (Feeder Analyst), the congestion penalty is dominant. 
            # We look for slots where baseline load is high, but capacity is not yet exceeded, 
            # or where adding our load keeps us safely under capacity while mitigating high carbon/price.
            
            if cost < min_cost:
                min_cost = cost
                best_slot = slot
                
        # Fallback/Tie-breaker if all costs were prohibitive or constraints failed (should not happen)
        if best_slot == -1:
            # Default to the slot with the lowest global carbon intensity if no feasible slot found
            day_data = self._get_day_data(day_index, day_name)
            best_slot = min(range(SLOT_COUNT), key=lambda i: day_data['carbon'][i])
            
        return best_slot

def load_scenario() -> Dict[str, Any]:
    """Loads the scenario.json file from the appropriate path."""
    # Assuming policy.py is executed from the directory containing scenario.json
    try:
        with open('scenario.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: scenario.json not found in the current directory.")
        exit(1)

def main():
    # 1. Load scenario data
    scenario_data = load_scenario()
    
    # Determine agent ID based on execution context (Assuming Agent 2 for this specific environment)
    AGENT_ID = 2 
    
    agent = EVPeakSharingAgent(scenario_data, AGENT_ID)
    
    recommendations = []
    day_names = list(scenario_data['days'].keys())
    
    # 2. Decide on a slot recommendation for each of the next seven days
    for day_index in range(agent.num_days):
        recommended_slot = agent.recommend_slot(day_index)
        recommendations.append(recommended_slot)

    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": AGENT_ID,
        "scenario_id": scenario_data["scenario_id"],
        "recommendations": [
            {"day_index": i, "slot_index": rec} 
            for i, rec in enumerate(recommendations)
        ]
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()