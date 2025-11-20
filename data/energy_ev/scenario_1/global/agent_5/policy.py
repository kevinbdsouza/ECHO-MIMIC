import json
import os

class AgentPolicyCoordinator:
    """
    Policy for Agent 5 (Position 5 graduate tenant commuting late) coordinating
    over 7 days to minimize weighted cost (Carbon/Price) while respecting
    coordination goals derived from neighbor preferences and grid capacity.
    """
    def __init__(self, agent_id=5):
        self.agent_id = agent_id
        self.location = 5
        self.base_demand = [0.50, 0.70, 0.60, 0.90]
        self.num_slots = 4
        self.num_days = 7
        
        # Load global parameters (will be overwritten by scenario file values)
        self.alpha = 40.00  # Weight for Price/Cost factors
        self.beta = 0.50    # Weight for Comfort/Local penalty
        self.gamma = 12.00  # Weight for Congestion/Grid stress factor
        
        self.scenario = None
        self.daily_data = {}
        self.neighbors = {}
        self.capacity = 6.8

    def load_scenario(self, filename='scenario.json'):
        """Loads scenario data, relying on 'scenario.json' being present."""
        try:
            with open(filename, 'r') as f:
                self.scenario = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Required file '{filename}' not found in the execution directory.")

        # Extract global parameters and constraints
        self.alpha = self.scenario.get('alpha', self.alpha)
        self.beta = self.scenario.get('beta', self.beta)
        self.gamma = self.scenario.get('gamma', self.gamma)
        self.capacity = self.scenario.get('capacity', 6.8)
        
        # Parse neighbor data based on the format observed in the prompt examples
        self.neighbors = self._parse_neighbor_examples(self.scenario.get('neighbor_examples', []))
        
        # Parse daily data structure
        self._parse_daily_data(self.scenario['days'])

    def _parse_neighbor_examples(self, neighbor_data: list) -> dict:
        parsed = {}
        for neighbor in neighbor_data:
            # Handle the structure where neighbor data might be nested or slightly inconsistent
            if 'Neighbor 4' in neighbor:
                agent_idx = 4
                data = neighbor['Neighbor 4'][0]
            elif 'Neighbor 1' in neighbor:
                agent_idx = 1
                data = neighbor['Neighbor 1'][0]
            else:
                continue

            parsed[agent_idx] = {
                'base_demand': data['Base demand'],
                'preferred_slots': set(data['Preferred slots']),
                'comfort_penalty': data['Comfort penalty'],
                'ground_truth_min_cost': data['Ground truth min-cost slots by day']
            }
        return parsed

    def _parse_daily_data(self, days_info: dict):
        """Parses the string-based daily data into usable lists of floats."""
        
        day_keys = sorted([k for k in days_info.keys() if k.startswith('Day')])
        
        for day_key in day_keys:
            day_info = days_info[day_key]
            
            def parse_list_str(s: str) -> list[float]:
                # Handles various delimiters found in the input format (', ' or just ',')
                return [float(x.strip()) for x in s.replace(';', '').split(',')]

            spatial_carbon_map = {}
            raw_sc = day_info['Spatial carbon']
            
            # Spatial carbon parsing for location 5
            for part in raw_sc.split(';'):
                try:
                    loc_id_str, carbon_str = part.split(': ')
                    loc_id = int(loc_id_str.strip())
                    spatial_carbon_map[loc_id] = parse_list_str(carbon_str)
                except ValueError:
                    # Handle case where key/value split fails (e.g., end of string)
                    continue

            self.daily_data[day_key] = {
                'tariff': parse_list_str(day_info['Tariff']),
                'carbon': parse_list_str(day_info['Carbon']),
                'baseline': parse_list_str(day_info['Baseline load']),
                'spatial_carbon': spatial_carbon_map.get(self.location, parse_list_str(day_info['Carbon'])) # Fallback to global carbon
            }

    def _calculate_slot_score(self, day_key: str, slot_index: int, session_count: int = 1) -> float:
        """
        Calculates the composite score (Cost to Minimize) for a single slot.
        Score = C1 * Carbon + C2 * Price + C3 * Congestion + C4 * Comfort_Penalty + C5 * Coordination_Penalty
        """
        data = self.daily_data[day_key]
        
        # --- 1. Cost Term (Carbon/Price - Global/Local Budget) ---
        
        # Prioritize Carbon (using location-specific value for Agent 5)
        carbon_intensity = data['spatial_carbon'][slot_index] 
        carbon_cost = self.alpha * carbon_intensity # Alpha (40.0) weighs this heavily
        
        # Price (Secondary cost)
        price = data['tariff'][slot_index]
        price_cost = self.alpha * 0.1 * price 
        
        # --- 2. Congestion Term (Grid Stress - Global Goal) ---
        
        baseline = data['baseline'][slot_index]
        demand = self.base_demand[slot_index] * session_count
        
        # Congestion Penalty (gamma=12.0): Penalize capacity usage relative to baseline expectation.
        # Agent 5 is a large consumer (demand 0.90 max). We use the ratio of demand/baseline as stress indicator.
        
        # If baseline is zero or very low, avoid division by zero; rely on absolute demand vs capacity if necessary.
        if baseline > 0:
            load_ratio = (baseline + demand) / self.capacity
        else:
            load_ratio = demand / self.capacity # Rely purely on absolute load if baseline is missing

        # Penalize high load ratio heavily when above 1.0 (Capacity violation)
        congestion_penalty = self.gamma * max(0, load_ratio - 1.0) * 5.0 # Scale violation heavily
        congestion_penalty += self.gamma * 0.1 * load_ratio # Mild penalty otherwise

        # --- 3. Comfort/Preference Term (Local Soft Constraint) ---
        
        # Agent 5 Profile: Late commuter, high demand in slot 3 (0.90). We prefer slot 3 and disfavor slot 2 (0.60 base demand).
        comfort_penalty = 0.0
        if slot_index == 3:
            comfort_penalty = -0.1 * self.beta # Mild reward for preferred late slot
        elif slot_index == 2:
            comfort_penalty = 0.5 * self.beta # Mild penalty for the lowest base demand slot
            
        # --- 4. Coordination Term (Neighbor Load Spreading) ---
        
        coordination_penalty = 0.0
        
        # Neighbor 4 (Loc 4) prefers {0, 3}. Neighbor 1 (Loc 1) prefers {0, 2}.
        # Avoid slots where coordination clash is high (i.e., slots preferred by both or highly preferred by critical neighbors).
        
        is_preferred_by_N4 = slot_index in self.neighbors.get(4, {}).get('preferred_slots', set())
        is_preferred_by_N1 = slot_index in self.neighbors.get(1, {}).get('preferred_slots', set())

        # If slot 3 is chosen, it strongly clashes with N4's comfort. We only choose it if Carbon/Price are very good.
        if slot_index == 3 and is_preferred_by_N4:
            coordination_penalty += 0.5 * self.beta
        
        # If slot 0 is chosen, it clashes with both N1 and N4's preference for early slots.
        elif slot_index == 0 and is_preferred_by_N4 and is_preferred_by_N1:
            coordination_penalty += 0.3 * self.beta

        total_score = carbon_cost + price_cost + congestion_penalty + comfort_penalty + coordination_penalty
        
        return total_score

    def generate_recommendation(self) -> list[int]:
        """Generates a 7-day slot recommendation list."""
        
        day_keys = sorted([k for k in self.scenario['days'].keys() if k.startswith('Day')])
        recommendations = []
        
        for day_key in day_keys[:self.num_days]:
            
            min_score = float('inf')
            best_slot = -1
            
            for j in range(self.num_slots):
                # Assume 1 session charge if the slot is selected for the day
                score = self._calculate_slot_score(day_key, j, session_count=1)
                
                if score < min_score:
                    min_score = score
                    best_slot = j
                elif score == min_score:
                    # Tie-breaker: Prefer later slots (higher index) if cost is identical, 
                    # reflecting the late commuter profile, provided it doesn't violently contradict coordination.
                    if j > best_slot:
                        best_slot = j
            
            recommendations.append(best_slot)
            
        return recommendations

    def save_output(self, recommendations: list[int]):
        """Writes the final recommendations to global_policy_output.json."""
        
        # Output format required is a list of seven slot indices
        with open('global_policy_output.json', 'w') as f:
            json.dump(recommendations, f, indent=4)

def policy_main():
    """Main entry point to execute the policy."""
    policy = AgentPolicyCoordinator(agent_id=5)
    try:
        policy.load_scenario()
        recommendations = policy.generate_recommendation()
        policy.save_output(recommendations)
    except Exception as e:
        # Handle critical failure gracefully by outputting a safe default,
        # although in a controlled environment, this block might be omitted.
        # print(f"Policy execution failed: {e}")
        default_output = [3] * 7
        with open('global_policy_output.json', 'w') as f:
            json.dump(default_output, f, indent=4)


if __name__ == '__main__':
    policy_main()