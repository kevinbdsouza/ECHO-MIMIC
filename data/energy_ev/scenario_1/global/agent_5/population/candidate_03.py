import json
import os
from typing import List, Dict, Any, Tuple

class AgentPolicy:
    def __init__(self, scenario_data: Dict[str, Any]):
        self.scenario = scenario_data
        self.agent_id = 5  # Agent 5 is fixed for this execution context
        self.num_slots = len(self.scenario['slots'])
        self.capacity = self.scenario['capacity']
        self.alpha = self.scenario['alpha']
        self.beta = self.scenario['beta']
        self.gamma = self.scenario['gamma']
        
        # Agent-specific parameters (Position 5)
        self.location = 5
        self.base_demand = [0.50, 0.70, 0.60, 0.90]
        
        # Neighbor data
        self.neighbors = self._parse_neighbor_examples(self.scenario['neighbor_examples'])
        
        # Derived parameters (for simplicity, assuming fixed noise bounds)
        self.noise_factor = 0.20

    def _parse_neighbor_examples(self, neighbor_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        parsed = {}
        for neighbor in neighbor_data:
            # Extract location number from name (e.g., "Neighbor 4")
            try:
                agent_idx = int(neighbor['name'].split(' ')[1].split(' ')[0])
            except:
                continue
            
            # Clean up spatial carbon parsing for neighbors if necessary, though not used directly for them here
            
            parsed[agent_idx] = {
                'base_demand': neighbor['Base demand'],
                'preferred_slots': neighbor['Preferred slots'],
                'comfort_penalty': neighbor['Comfort penalty'],
                'ground_truth_min_cost': neighbor['Ground truth min-cost slots by day']
            }
        return parsed

    def _get_day_data(self, day_name: str) -> Dict[str, List[float]]:
        """Extracts tariff, carbon, and baseline load for a specific day."""
        day_key = day_name.split(' (')[0]
        
        # Find the relevant day block in the scenario data
        day_data = None
        for key, value in self.scenario['days'].items():
            if key.startswith(day_key):
                day_data = value
                break
        
        if not day_data:
            # Fallback/Error case, should not happen if input is valid
            return {
                'tariff': self.scenario['price'],
                'carbon': self.scenario['carbon_intensity'],
                'baseline_load': self.scenario['baseline_load']
            }

        # Parse the string lists into floats
        def parse_list_str(s: str) -> List[float]:
            return [float(x) for x in s.split(', ')]

        # Spatial carbon needs special handling if used, but we focus on main metrics first
        
        return {
            'tariff': parse_list_str(day_data['Tariff']),
            'carbon': parse_list_str(day_data['Carbon']),
            'baseline_load': parse_list_str(day_data['Baseline load']),
            'spatial_carbon': day_data['Spatial carbon']
        }

    def _calculate_slot_scores(self, day_index: int, day_key: str, data: Dict[str, List[float]]) -> List[Tuple[float, int]]:
        """
        Calculates a composite score for each slot based on cost, carbon, congestion, 
        and neighborhood awareness. Lower score is better.
        """
        scores = []
        
        # 1. Extract spatial carbon for *this* agent's location (Location 5)
        spatial_carbon_map = {}
        spatial_data_raw = data['spatial_carbon']
        
        try:
            # Spatial carbon format: "1: 330, 520, 560, 610; 2: ..."
            parts = spatial_data_raw.split(';')
            for part in parts:
                loc_id_str, carbon_str = part.split(': ')
                loc_id = int(loc_id_str.strip())
                spatial_carbon_map[loc_id] = [float(c) for c in carbon_str.strip().split(', ')]
        except Exception as e:
            # Fallback if parsing fails: assume uniform/use global carbon only
            print(f"Warning: Failed to parse spatial carbon for {day_key}. Error: {e}")
            pass

        agent_spatial_carbon = spatial_carbon_map.get(self.location, data['carbon']) # Use global if specific is missing

        # 2. Determine Neighbor Load Awareness (Coordination Goal)
        # We look at the *preferred* slots of neighbors. High preference density suggests high local demand.
        neighbor_preference_counts = [0] * self.num_slots
        for nid, n_data in self.neighbors.items():
            for slot in n_data['preferred_slots']:
                neighbor_preference_counts[slot] += 1
        
        # Heuristic: Penalize slots heavily favored by neighbors, especially if they are trying to shift load.
        # Since neighbors 1 and 4 prefer early (0, 2) and late (3), we weigh coordination carefully.
        
        # 3. Iterate through slots to calculate scores
        for j in range(self.num_slots):
            
            # Base Load + Noise Consideration
            # Assume an average load projection that includes neighborhood baseline shift (approximated)
            # Since we don't know neighbor *actual* load, we use the perceived congestion from baselines + our own demand.
            
            # Personal Load Estimate (including noise uncertainty)
            demand_j = self.base_demand[j]
            
            # Congestion/Transformer Constraint (Global Goal: Capacity Limit)
            # We don't know the *total* load yet, but we aim to keep our contribution moderate if others are already high.
            # Here we use the aggregate baseline load as a proxy for grid stress.
            aggregate_baseline = data['baseline_load'][j]
            
            # Cost and Carbon (Primary minimization factors)
            price_j = data['tariff'][j]
            carbon_j = data['carbon'][j]
            spatial_carbon_j = agent_spatial_carbon[j] # Use location-specific carbon if available

            # --- Cost Function Components ---
            
            # 1. Comfort (Personal Constraint)
            # Agent 5 is a late commuter, prioritizing later slots implicitly by higher base demand there (0.90 at slot 3).
            # Lower base demand implies lower personal cost/preference for that slot.
            # We invert the base demand: lower demand means higher comfort penalty to encourage use.
            # However, the prompt suggests Position 5 commutes late, meaning high demand slots (3) are likely *necessary*.
            # Let's use a simple penalty based on deviation from the *ideal* (which we assume is slot 3 based on demand profile)
            
            comfort_penalty_factor = 0.0
            if j == 3:
                comfort_penalty_factor = 0.0 # Preferred slot based on demand profile
            elif j == 0:
                comfort_penalty_factor = 0.3  # Least preferred slot
            else:
                comfort_penalty_factor = 0.15

            # 2. Carbon Cost (Global Goal 1) - Weighted by Gamma
            carbon_cost = self.gamma * spatial_carbon_j

            # 3. Price Cost (Personal Budget) - Weighted by Alpha
            price_cost = self.alpha * price_j

            # 4. Congestion Cost (Global Goal 2) - Weighted by Beta, scaled by perceived grid stress
            # We penalize slots where neighbors are highly concentrated, scaled by how stressed the grid already seems (aggregate baseline).
            congestion_proxy = aggregate_baseline / self.capacity
            neighbor_coordination_penalty = self.beta * neighbor_preference_counts[j] * congestion_proxy
            
            # --- Final Score Calculation ---
            # Score = Alpha*Price + Gamma*Carbon + Beta*Coordination + Comfort_Factor
            
            score = (
                price_cost + 
                carbon_cost + 
                neighbor_coordination_penalty +
                comfort_penalty_factor
            )
            
            # Add slot index for tracking
            scores.append((score, j))
            
        # Sort by score (ascending)
        scores.sort(key=lambda x: x[0])
        
        # Filter based on min/max session constraints (If we were scheduling, we'd check, but for recommendation, we only score)
        # Assuming constraints are soft for this recommendation step, but we must respect them if they lead to infeasibility.
        # Since we are only recommending *one* slot, we assume this slot will be used.
        
        return scores

    def generate_recommendation(self) -> List[int]:
        """Generates a 7-day slot recommendation."""
        
        day_names = list(self.scenario['days'].keys())
        recommendations = []
        
        # The scenario provides 7 days starting from Day 1
        for day_index, day_name in enumerate(day_names[:7]):
            
            # 1. Get day-specific data (Tariff, Carbon, Baseline, Spatial Carbon)
            day_data = self._get_day_data(day_name)
            
            # 2. Calculate scores for all 4 slots
            scored_slots = self._calculate_slot_scores(day_index, day_name, day_data)
            
            # 3. Select the best slot (lowest score)
            best_score, best_slot_index = scored_slots[0]
            
            # 4. Apply simple heuristic adjustments based on neighbor coordination goals
            # Neighbor 4 (Retirees, Loc 4) often prefers slot 3 (late) for comfort.
            # Neighbor 1 (Battery Eng, Loc 1) prefers slots 0 and 2 (flexible).
            
            # Agent 5 (Pos 5, Commuter) has high demand late (slot 3: 0.90).
            # Coordination Goal: If the grid is stressed (high aggregate baseline), try to shift away from neighbor hotspots IF comfort allows.
            
            # Check if the best slot aligns with Neighbor 4's preferred slots (0 or 3) or Neighbor 1's (0 or 2)
            # Since Agent 5 has high demand at slot 3, we strongly favor it unless it is critically bad for carbon.
            
            final_slot = best_slot_index
            
            # Specific Day Check: Day 2 notes slots 0 and 3 must balance transformer temps.
            # If the calculated best slot is 0 or 3, we stick to it, as the score already factored in the stress.
            
            # Specific Day Check: Day 4 notes neighborhood watch enforces staggered use.
            # If the best slot is one used heavily by neighbors (e.g., slot 3 for N4), maybe shift slightly if score difference is small.
            
            # Given the strong weighting (alpha=40, gamma=12) on Price and Carbon, the calculated best_slot is usually optimal
            # for minimizing the weighted sum, which captures global goals (Carbon, Congestion proxy).
            
            recommendations.append(final_slot)
            
        return recommendations

    def save_output(self, recommendations: List[int]):
        """Saves the recommendations to global_policy_output.json."""
        output_data = {
            "agent_id": self.agent_id,
            "location": self.location,
            "recommendations": [
                {"day": i + 1, "slot": rec} for i, rec in enumerate(recommendations)
            ]
        }
        
        # The specification requires a list of seven slot indices directly
        final_list = recommendations

        with open("global_policy_output.json", "w") as f:
            json.dump(final_list, f, indent=4)

def main():
    # Load scenario data from scenario.json in the current directory
    try:
        with open("scenario.json", 'r') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        # If running in an environment where the file structure is relative, adjust path if necessary.
        # Assuming 'scenario.json' is present in the execution directory.
        print("Error: scenario.json not found.")
        return

    policy_solver = AgentPolicy(scenario_data)
    recommendations = policy_solver.generate_recommendation()
    policy_solver.save_output(recommendations)

if __name__ == "__main__":
    main()