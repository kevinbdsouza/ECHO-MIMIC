import json
import os
from typing import List, Dict, Any, Tuple

class Policy:
    def __init__(self, scenario_data: Dict[str, Any], agent_id: int):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.slot_indices = list(range(len(self.scenario['slots'])))
        
        # Agent-specific configuration extraction
        self.location_idx = self.scenario['agent_id_to_location'][str(agent_id)]
        
        # Base demand (kW) for this agent across the 4 slots
        self.base_demand = self.scenario['base_demand']
        
        # Coordination parameters
        self.alpha = self.scenario['alpha']  # Carbon/Price sensitivity
        self.beta = self.scenario['beta']    # Neighbor interaction weight
        self.gamma = self.scenario['gamma']  # Local comfort/Capacity sensitivity

        # Neighbor data (Assuming we can access neighbor examples based on location/ID proximity)
        self.neighbors = self._parse_neighbor_examples(self.scenario['neighbor_examples'])

        # Scenario context (for reference, though this policy focuses on lookahead)
        self.scenario_context = {
            'capacity': self.scenario['capacity'],
            'time_prices': self.scenario['price'],
            'time_carbon': self.scenario['carbon_intensity']
        }

    def _parse_neighbor_examples(self, neighbor_examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        parsed = {}
        for ex in neighbor_examples:
            name_parts = ex['Position'].split()
            parsed[name_parts[1]] = ex  # Key by position number e.g., '3', '5'
        return parsed

    def _get_day_data(self, day_name: str) -> Dict[str, List[float]]:
        """Extracts tariff, carbon, baseline load, and spatial carbon for a specific day."""
        day_data = {}
        # Find the matching day entry in the scenario data
        target_day = next(
            (day for day in self.scenario['days'] if day_name in day), None
        )
        
        if not target_day:
            raise ValueError(f"Could not find data for {day_name}")

        day_key = list(target_day.keys())[0]
        data = target_day[day_key]
        
        # Parse lists from strings
        day_data['tariff'] = [float(x) for x in data['Tariff'].split(', ')]
        day_data['carbon'] = [float(x) for x in data['Carbon'].split(', ')]
        day_data['baseline_load'] = [float(x) for x in data['Baseline load'].split(', ')]
        
        # Spatial carbon parsing: We need the data for *our* location index
        spatial_carbon_str = data['Spatial carbon'].split(';')[self.location_idx - 1]
        # spatial_carbon_str format: "1: 330, 520, 560, 610"
        
        # Extract the 4 slot values relevant to this agent's location
        slot_values_str = spatial_carbon_str.split(': ')[1]
        day_data['spatial_carbon'] = [float(x) for x in slot_values_str.split(', ')]
        
        return day_data

    def _calculate_slot_score(self, day_idx: int, slot_idx: int, day_data: Dict[str, List[float]]) -> float:
        """
        Calculates a combined cost/benefit score for a specific slot on a specific day.
        Lower score is better (lower cost/higher benefit).
        
        Score = alpha * Normalized_Cost + gamma * Normalized_Load_Impact + beta * Neighbor_Coordination
        """
        
        # --- 1. Local Objective (Cost Minimization) ---
        
        # Cost components (Price and Carbon Intensity for the day/slot)
        price = day_data['tariff'][slot_idx]
        carbon = day_data['carbon'][slot_idx]
        
        # Normalize Cost using scenario context for relative comparison across days
        norm_price = price / self.scenario_context['time_prices'][slot_idx]
        norm_carbon = carbon / self.scenario_context['time_carbon'][slot_idx]
        
        cost_score = (norm_price + norm_carbon) / 2.0

        # --- 2. Local Objective (Congestion/Comfort) ---
        
        # Capacity constraint based score (Penalize high local use relative to baseline/capacity)
        capacity = self.scenario_context['capacity']
        baseline = day_data['baseline_load'][slot_idx]
        agent_load = self.base_demand[slot_idx]
        spatial_congestion = day_data['spatial_carbon'][slot_idx]
        
        # Heuristic for local impact: Combine agent's own load with neighbor spatial impact
        # High spatial carbon implies high congestion or dirty power source at that location/time.
        
        # Load impact: Penalize slots where baseline + agent load is close to capacity OR spatial carbon is high
        load_factor = (baseline + agent_load) / capacity
        
        # Penalize high spatial congestion severely (as per persona: comfort/grid warnings)
        # We scale spatial congestion based on the maximum observed spatial carbon in the scenario context, 
        # although using the day's max is more dynamic. Let's use the day's max for normalization against other slots *on that day*.
        max_spatial_carbon_day = max(day_data['spatial_carbon'])
        norm_spatial_carbon = spatial_carbon / max_spatial_carbon_day if max_spatial_carbon_day > 0 else 0

        # Gamma heavily weights local comfort/congestion factors
        congestion_score = (load_factor * 0.5) + (norm_spatial_carbon * 0.5)

        # --- 3. Coordination Objective ---
        
        # Analyze neighbor preferences. We want to avoid their preferred slots if they conflict with our local goals, 
        # OR we can aim to align if they are trying to solve a major global issue (like lowest carbon).
        
        neighbor_penalty = 0.0
        
        # Look at explicit neighbor preferences (assuming the agent knows neighbor profiles from history/context)
        for neighbor_id, n_data in self.neighbors.items():
            pref_slots = n_data.get('Preferred slots', [])
            
            if slot_idx in pref_slots:
                # If slot is preferred by a neighbor, apply a small adjustment (neutralizing penalty for simplicity in this lookahead)
                # For Stage 3 collective, the goal is *not* to rigidly avoid, but to be aware.
                # If neighbors prefer low-cost slots, this might increase our cost. We apply a mild cost offset.
                neighbor_penalty += self.beta * 0.1
        
        # Look at actual historical behavior (Ground Truth)
        # If a neighbor historically chose a slot, this slot might be "important" for them or the grid state they anticipate.
        # Penalize slots that neighbors frequently chose (as they might be draining a shared resource or capacity).
        
        historical_weight = 0.0
        for neighbor_id, n_data in self.neighbors.items():
            # We need the recommendation for this specific day (Day index 0-6 maps to Day 1-7)
            gt_key = f"Day {day_idx + 1}"
            if gt_key in n_data.get('Ground truth min-cost slots by day', {}):
                if slot_idx in n_data['Ground truth min-cost slots by day'][gt_key]:
                    historical_weight += self.beta * 0.2 # Slightly higher weight for observed past behavior

        neighbor_penalty += historical_weight
        
        # --- Final Score ---
        
        # Weights: Alpha for environment cost, Gamma for local congestion, Beta for social awareness
        final_score = (
            self.alpha * cost_score +
            self.gamma * congestion_score +
            neighbor_penalty
        )
        
        # Apply slot constraints (Min/Max Sessions are usually handled during scheduling, but here we use them as soft constraints via high penalty)
        min_sessions = self.scenario['slot_min_sessions'][f'slot_{slot_idx}']
        max_sessions = self.scenario['slot_max_sessions'][f'slot_{slot_idx}']

        # Since we are only recommending *one* slot per day, we only use min/max sessions for context, 
        # assuming the scheduler will check these against the 7-day plan later. For scoring one day, we focus on cost/comfort.

        return final_score

    def recommend_slots(self) -> List[int]:
        """Calculates the recommended slot for each of the 7 days."""
        
        recommendations = []
        
        # Iterate through Day 1 to Day 7 (indices 0 to 6)
        day_names = [f"Day {i+1} ({self.scenario['days'][i].get(list(self.scenario['days'][i].keys())[0]).split(')')[0].split(' — ')[1]})" 
                     for i in range(7)]

        for day_idx, day_name in enumerate(day_names):
            try:
                day_data = self._get_day_data(day_name)
            except ValueError as e:
                print(f"Error processing {day_name}: {e}")
                # Fallback: Use scenario context averages if day data is missing
                day_data = {
                    'tariff': self.scenario_context['time_prices'],
                    'carbon': self.scenario_context['time_carbon'],
                    'baseline_load': self.scenario['baseline_load'],
                    'spatial_carbon': [0.0] * 4 # Cannot calculate without specific spatial data
                }
                # If data fails, rely heavily on global time prices/carbon
                
            
            best_slot = -1
            min_score = float('inf')
            
            slot_scores = {}
            
            for slot_idx in self.slot_indices:
                score = self._calculate_slot_score(day_idx, slot_idx, day_data)
                slot_scores[slot_idx] = score
                
                if score < min_score:
                    min_score = score
                    best_slot = slot_idx
            
            # Post-selection check: Ensure we didn't pick an impossible slot (although constraints are soft here)
            if best_slot == -1:
                # Should not happen if slots are 0-3, but as safety, pick the cheapest slot based on scenario context price
                default_slot = min(enumerate(self.scenario_context['time_prices']), key=lambda x: x[1])[0]
                best_slot = default_slot

            recommendations.append(best_slot)

        return recommendations

def main():
    # 1. Load scenario data
    try:
        # Assuming policy.py is run from the agent's directory containing scenario.json
        with open('scenario.json', 'r') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        print("Error: scenario.json not found in the current directory.")
        return

    # Determine Agent ID (Crucial for locating agent-specific data)
    # In a real system, this might be an environment variable. Here we deduce it 
    # based on the common structure where Agent 4 corresponds to position 4 data.
    # We check the scenario to find the mapping, but since the prompt specifies Agent 4, we hardcode for safety.
    AGENT_ID = 4
    
    # 2. Decide on slot recommendations
    policy = Policy(scenario_data, AGENT_ID)
    recommendations = policy.recommend_slots()
    
    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": AGENT_ID,
        "location": policy.location_idx,
        "recommendations": [
            {"day": i + 1, "slot": rec} for i, rec in enumerate(recommendations)
        ]
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()