import json
import os
from typing import List, Dict, Any

class Policy:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.scenario_data: Dict[str, Any] = {}
        self.params: Dict[str, Any] = {}
        self.slot_info: Dict[str, Any] = {}
        self.days_data: Dict[str, Any] = {}
        self.neighbor_examples: List[Dict[str, Any]] = []
        self.load_scenario()

    def load_scenario(self):
        # Construct the path assuming policy.py is run from the agent directory
        try:
            with open('scenario.json', 'r') as f:
                self.scenario_data = json.load(f)
        except FileNotFoundError:
            # Fallback or error handling if running outside the expected structure
            print("Error: scenario.json not found. Ensure the script is run from the agent directory.")
            return

        # Extract key components
        self.slot_info = {
            "slots": self.scenario_data["slots"],
            "price": self.scenario_data["price"],
            "carbon_intensity": self.scenario_data["carbon_intensity"],
            "capacity": self.scenario_data["capacity"],
            "baseline_load": self.scenario_data["baseline_load"],
            "slot_min_sessions": self.scenario_data["slot_min_sessions"],
            "slot_max_sessions": self.scenario_data["slot_max_sessions"],
        }
        self.days_data = self.scenario_data["days"]
        self.neighbor_examples = self.scenario_data.get("neighbor_examples", [])
        self.alpha = self.scenario_data["alpha"]
        self.beta = self.scenario_data["beta"]
        self.gamma = self.scenario_data["gamma"]
        self.persona = self.scenario_data["persona"]
        self.location = self.scenario_data["location"]
        self.base_demand = self.scenario_data["base_demand"]

    def _parse_spatial_carbon(self, spatial_str: str, slot_indices: List[int]) -> Dict[int, float]:
        # Maps spatial carbon string to a dictionary keyed by location ID
        parts = spatial_str.split(';')
        sc_data = {}
        for part in parts:
            try:
                loc_id_str, values_str = part.split(':')
                loc_id = int(loc_id_str.strip())
                values = [float(v.strip()) for v in values_str.split(',')]
                # Assuming values match the order of slots 0, 1, 2, 3
                for i, slot_idx in enumerate(slot_indices):
                    sc_data[(loc_id, slot_idx)] = values[i]
            except ValueError:
                continue
        return sc_data

    def calculate_cost(self, day_key: str, slot_index: int, day_data: Dict[str, Any]) -> float:
        slot_indices = list(range(len(self.slot_info['slots'])))

        # 1. Personal Cost (Comfort/Demand)
        # Agent 4: Position 4 retirees guarding comfort. High focus on comfort.
        # Base demand: 0.90, 0.60, 0.70, 0.80 (High demand in slot 0)
        # Assume comfort penalty is high for deviating from base demand pattern.
        
        comfort_penalty = self.gamma * abs(self.base_demand[slot_index] - self.slot_info["baseline_load"][slot_index] / 10.0) # Scaling baseline load for comparison

        # 2. Global Cost (Carbon/Price)
        # Alpha * Carbon + Beta * Price
        
        tariff = day_data["Tariff"][slot_index]
        carbon_intensity = day_data["Carbon"][slot_index]
        
        environmental_cost = self.alpha * carbon_intensity + self.beta * tariff

        # 3. Spatial Cost (Congestion/Neighborhood)
        # We need to calculate the average spatial carbon for *this* agent's location (4)
        spatial_carbon_data = self._parse_spatial_carbon(day_data["Spatial carbon"], slot_indices)
        
        # The agent is at location 4
        location_carbon = spatial_carbon_data.get((self.location, slot_index), float('inf'))
        
        # Given the prompt context, spatial carbon often represents localized congestion impact.
        # Penalize high spatial carbon at the agent's location.
        spatial_cost = self.gamma * (location_carbon / 1000.0) # Normalize relative to other factors

        # Combine Costs
        # Given the persona (retirees guarding comfort), comfort/local factors (gamma) should be influential,
        # but coordination goals (alpha, beta) must still be met.
        
        total_cost = environmental_cost + comfort_penalty + spatial_cost
        
        # Apply neighbor preference hints (if we could observe current neighbor demand, we would shift)
        # Since we only have *examples*, we use them to confirm our general strategy:
        # Neighbor 3 (nurse, loc 3) prefers 1, 3. Neighbor 5 (commuter, loc 5) prefers 0, 1.
        # Agent 4 prefers high demand early (Slot 0: 0.9). We will prioritize low cost heavily, 
        # but slightly favor slots where neighbors *aren't* heavily concentrated if costs are equal.
        
        return total_cost

    def get_recommendation(self) -> List[int]:
        recommendations = []
        day_keys = list(self.days_data.keys())
        
        # Limit to 7 days if more are present, or use all available day keys
        days_to_process = day_keys[:7] 

        for day_key in days_to_process:
            day_data = self.days_data[day_key]
            
            # Parse spatial carbon data for the current day once
            slot_indices = list(range(len(self.slot_info['slots'])))
            
            # The baseline carbon intensity for global calculation uses the scenario's default slots
            # For day-specific carbon calculation, we use the day's specific data.
            
            best_slot = -1
            min_cost = float('inf')
            
            # Constraints check (Min/Max Sessions are usually for scheduling multiple sessions, 
            # but here we assume the agent only picks ONE slot for the entire day)
            min_sessions = self.slot_info["slot_min_sessions"]
            max_sessions = self.slot_info["slot_max_sessions"]
            
            for i in slot_indices:
                # Check basic feasibility (assuming we must meet min sessions if that implies being selected)
                # Since this is a 1-slot recommendation per day, we just check against max sessions (if >= 1, it's possible)
                if max_sessions[i] < 1:
                    continue

                cost = self.calculate_cost(day_key, i, day_data)
                
                if cost < min_cost:
                    min_cost = cost
                    best_slot = i
            
            # Fallback: If no slot was selected (e.g., all blocked by capacity/min_sessions, though unlikely here)
            if best_slot == -1:
                # Default to the cheapest slot overall based on static scenario data if logic fails
                static_costs = [
                    self.alpha * self.slot_info['carbon_intensity'][i] + self.beta * self.slot_info['price'][i]
                    for i in slot_indices
                ]
                best_slot = static_costs.index(min(static_costs))


            recommendations.append(best_slot)

        return recommendations

def main():
    # Agent ID is hardcoded as 4 based on the prompt context
    agent_id = 4
    policy = Policy(agent_id)
    recommendations = policy.get_recommendation()

    # Prepare output structure
    # The output must be a list of seven slot indices
    global_output = {
        "agent_id": agent_id,
        "recommendations": recommendations
    }

    # Save global_policy_output.json
    output_filename = 'global_policy_output.json'
    with open(output_filename, 'w') as f:
        json.dump(global_output, f, indent=4)

    # The script is self-contained and executable as policy.py
    # No further output is required besides the file creation.

if __name__ == "__main__":
    main()
# End of policy.py