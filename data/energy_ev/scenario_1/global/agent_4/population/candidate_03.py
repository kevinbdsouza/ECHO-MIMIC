import json
import os

class AgentPolicy:
    def __init__(self, scenario_data, agent_id):
        self.scenario = scenario_data
        self.agent_id = agent_id
        
        # Agent-specific parameters derived from the prompt
        self.location = self.scenario['location']
        self.base_demand = self.scenario['base_demand']
        self.alpha = self.scenario['alpha']
        self.beta = self.scenario['beta']
        self.gamma = self.scenario['gamma']
        
        # System parameters
        self.num_slots = len(self.scenario['slots'])
        self.neighbor_examples = self.scenario['neighbor_examples']
        self.days_data = self.scenario['days']
        self.day_names = list(self.days_data.keys())

    def _parse_spatial_carbon(self, spatial_carbon_str, day_index):
        """Parses the spatial carbon string for a specific day index."""
        try:
            # Find the relevant spatial carbon data for the day
            day_key = [k for k in self.days_data.keys() if f"Day {day_index + 1}" in k][0]
            
            # The structure is: LocationX: C1, C2, C3, C4; LocationY: C1, C2, C3, C4
            # We need the values associated with our agent's location
            
            # Extract all location data strings
            location_data_map = {}
            parts = spatial_carbon_str.split(';')
            for part in parts:
                loc, carbons = part.strip().split(':')
                location_data_map[int(loc.strip())] = [float(c.strip()) for c in carbons.split(',')]

            if self.location in location_data_map:
                return location_data_map[self.location][day_index]
            else:
                # Should not happen if input is correct, use an average or raise error
                return 500.0 # Default fallback
        except Exception as e:
            # print(f"Error parsing spatial carbon for day {day_index}: {e}")
            return 500.0 # Fallback

    def calculate_cost(self, day_index, slot_index, num_neighbors_sharing):
        """
        Calculates the composite cost for choosing a specific slot on a given day.
        Cost = Alpha * Local_Comfort_Cost + Beta * Global_Carbon_Cost + Gamma * Congestion_Cost
        """
        day_key = self.day_names[day_index]
        day_info = self.days_data[day_key]
        
        # 1. Local Comfort Cost (Inverse of utility/satisfaction)
        # Comfort is generally related to deviation from baseline demand. 
        # Since we don't have a direct comfort function, we use the penalty derived from proximity 
        # to the base demand (lower demand satisfaction cost implies higher comfort, but here we minimize cost).
        # We will use the baseline load deviation as a proxy for difficulty/discomfort if we must deviate significantly.
        # Since this agent cares about comfort (Position 4 retirees), we penalize deviations from baseline.
        
        baseline = day_info['Baseline load'][slot_index]
        demand = self.base_demand[slot_index]
        
        # A simple comfort penalty: deviation magnitude squared, scaled by baseline demand (more load = higher impact)
        comfort_cost = abs(demand - baseline) * 10.0 
        
        # 2. Global Carbon Cost
        # Use the average carbon intensity for the day/slot, incorporating spatial carbon if necessary.
        # Carbon data provided is day-specific (Carbon list) and location-specific (Spatial carbon list).
        
        day_carbon = day_info['Carbon'][slot_index]
        
        # Calculate spatial carbon impact for this agent's location in this slot
        spatial_carbon_str = day_info['Spatial carbon']
        spatial_carbon = self._parse_spatial_carbon(spatial_carbon_str, slot_index)
        
        # Weighted carbon cost: heavily weight the local grid carbon intensity
        carbon_cost = day_carbon + 0.5 * spatial_carbon

        # 3. Congestion Cost (Transformer/Capacity Constraint)
        # Congestion is driven by how many neighbors are also scheduled, relative to the capacity limit.
        # Capacity is 6.8.
        
        # For this agent, we estimate its demand contribution based on its base demand relative to others.
        # Let's assume our agent demands self.base_demand[slot_index] * 1 unit of load if scheduled.
        
        # The congestion cost must penalize being too high above capacity, weighted by how many others share the slot.
        
        # Assume all other neighbors in the same slot contribute an estimated load (using their base demand average for simplicity)
        # Neighbor contributions (simplified average load estimation for coordination)
        neighbor_loads = []
        for neighbor_data in self.neighbor_examples:
            n_base_demand = neighbor_data['Base demand']
            # Assuming neighbor demand is also scaled by their slot demand ratio
            neighbor_loads.append(n_base_demand[slot_index])
            
        total_estimated_load = self.base_demand[slot_index] + sum(neighbor_loads)
        
        capacity = self.scenario['capacity']
        
        # Penalize exceeding capacity, scaled by the number of users sharing (num_neighbors_sharing includes this agent)
        congestion_penalty = 0
        if total_estimated_load > capacity:
            overload = total_estimated_load - capacity
            # Stronger penalty if more people are sharing (higher num_neighbors_sharing)
            congestion_penalty = overload * 5.0 * num_neighbors_sharing 
        
        # 4. Tariff/Price Cost (Slightly lower weight than carbon/comfort, but important)
        tariff = day_info['Tariff'][slot_index]
        price_cost = tariff * 5.0

        # Final Composite Cost
        total_cost = (self.alpha * comfort_cost) + \
                     (self.beta * carbon_cost) + \
                     (self.gamma * congestion_penalty) + \
                     (0.1 * price_cost)
                     
        return total_cost, {'comfort': comfort_cost, 'carbon': carbon_cost, 'congestion': congestion_penalty, 'price': price_cost}

    def get_neighbor_schedule(self, day_index):
        """
        Predicts neighbor schedules for the current day based on their historical minimum cost choices,
        respecting their slot constraints (min/max sessions).
        This is a simplified model reflecting observed behavior.
        """
        day_key = self.day_names[day_index]
        
        neighbor_schedules = {}
        
        # Get slot constraints for the day (These are system-wide, but we check neighbor minimums/maximums implicitly later)
        slot_min_sessions = self.scenario['slot_min_sessions']
        slot_max_sessions = self.scenario['slot_max_sessions']
        
        for neighbor_name, neighbor_data in self.neighbor_examples.items():
            # Use the provided ground truth as the deterministic prediction for coordination context
            if day_index < len(neighbor_data['Ground truth min-cost slots by day']):
                predicted_slot = neighbor_data['Ground truth min-cost slots by day'][day_index]
                neighbor_schedules[neighbor_name] = predicted_slot
            else:
                # Fallback: If history is missing for a future day, assume a neutral slot or their preferred slot
                neighbor_schedules[neighbor_name] = neighbor_data.get('Preferred slots', [1])[0] 
                
        return neighbor_schedules

    def coordinate_and_recommend(self):
        recommendations = []
        
        # The agent must make a 7-day schedule recommendation
        for day_index in range(7):
            day_key = self.day_names[day_index]
            
            # 1. Determine neighbor usage for this day (Coordination context)
            neighbor_schedule = self.get_neighbor_schedule(day_index)
            
            # 2. Evaluate all slots for this day
            slot_costs = []
            
            for slot_index in range(self.num_slots):
                
                # Count how many neighbors are sharing this slot
                num_neighbors_sharing = sum(1 for slot in neighbor_schedule.values() if slot == slot_index)
                
                # Add self to the count if we decide to schedule here (we calculate cost assuming we *do* schedule)
                # For cost calculation, we need to include ourselves in the congestion calculation
                
                cost, details = self.calculate_cost(day_index, slot_index, num_neighbors_sharing + 1)
                
                # Add constraints check (If this agent MUST run minimum, we heavily penalize slots that violate it)
                # For simplicity in this stage, we assume the constraints are soft unless explicitly told otherwise,
                # but we need to respect the min/max session counts if we were aggregating the whole group's schedule.
                # Since we only output one slot, we assume this single selection satisfies minimum session needs locally.
                
                slot_costs.append({
                    'slot': slot_index,
                    'cost': cost,
                    'details': details,
                    'neighbor_count': num_neighbors_sharing
                })

            # 3. Select the best slot (Minimum Cost)
            best_slot_info = min(slot_costs, key=lambda x: x['cost'])
            
            recommended_slot = best_slot_info['slot']
            recommendations.append(recommended_slot)
            
            # Logging/Debugging (optional but helpful for local reasoning)
            # print(f"{day_key} (Loc {self.location}): Recommended Slot {recommended_slot} (Cost: {best_slot_info['cost']:.2f})")
            # print(f"  Neighbor count in slot: {best_slot_info['neighbor_count']}")
            # print(f"  Costs Breakdown: {best_slot_info['details']}")

        return recommendations

def main():
    # 1. Load scenario.json
    try:
        # Assuming policy.py is run from the agent's directory which contains scenario.json
        with open('scenario.json', 'r') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        # Handle case where file path might need adjustment based on execution environment
        # For this exercise, we assume direct access relative to the script location.
        # If running in a container/test harness, this path must be correct.
        raise FileNotFoundError("scenario.json not found in the expected location.")

    # Agent 4 identification is hardcoded in the prompt context
    agent_id = 4
    
    policy_solver = AgentPolicy(scenario_data, agent_id)
    
    # 2. Decide on the slot recommendation
    recommendations = policy_solver.coordinate_and_recommend()
    
    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": agent_id,
        "recommendations": recommendations
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()