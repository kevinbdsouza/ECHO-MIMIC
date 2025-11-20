import json
import os

# Define the path to scenario.json based on the agent's directory structure assumption
# Assuming the policy.py is executed from a directory where scenario.json is one level up
# or in a standardized location relative to the script execution environment.
# For this specific environment, we assume direct loading is possible if the file is present.
SCENARIO_FILE = 'scenario.json'

class EVSchedulingPolicy:
    def __init__(self, agent_id=5):
        self.agent_id = agent_id
        self.num_slots = 4
        self.num_days = 7
        self.base_demand = [0.50, 0.70, 0.60, 0.90]
        self.location = 5
        self.alpha = 40.00  # Weight for cost/carbon
        self.beta = 0.50    # Weight for comfort penalty (local)
        self.gamma = 12.00  # Weight for congestion/baseline adherence

        # Neighbor specific data (Hardcoded based on prompt context)
        self.neighbor_data = {
            4: {'location': 4, 'base_demand': [0.90, 0.60, 0.70, 0.80], 'preferred': [0, 3], 'comfort_penalty': 0.16},
            1: {'location': 1, 'base_demand': [1.20, 0.70, 0.80, 0.60], 'preferred': [0, 2], 'comfort_penalty': 0.18}
        }

    def load_scenario(self, filename=SCENARIO_FILE):
        """Loads scenario data from a JSON file."""
        try:
            with open(filename, 'r') as f:
                scenario = json.load(f)
        except FileNotFoundError:
            # Handle case where running outside the expected directory structure
            # If this happens, we'll use the hardcoded structure from the prompt context for slots/params
            print(f"Warning: {filename} not found. Using structure defined in prompt.")
            scenario = self._get_hardcoded_structure()
            
        self.scenario = scenario
        self._extract_agent_specific_data(scenario)
        return scenario

    def _get_hardcoded_structure(self):
        """Creates a minimal structure if JSON loading fails, based on prompt context."""
        return {
            "slots": {
                "0": "19-20", "1": "20-21", "2": "21-22", "3": "22-23"
            },
            "price": [0.23, 0.24, 0.27, 0.30],
            "carbon_intensity": [700, 480, 500, 750],
            "capacity": 6.8,
            "baseline_load": [5.2, 5.0, 4.9, 6.5],
            "slot_min_sessions": {"0": 1, "1": 1, "2": 1, "3": 1},
            "slot_max_sessions": {"0": 2, "1": 2, "2": 1, "3": 2},
            "spatial_carbon": {
                "1": "440, 460, 490, 604", "2": "483, 431, 471, 600", "3": "503, 473, 471, 577",
                "4": "617, 549, 479, 363", "5": "411, 376, 554, 623"
            },
            "days": {
                "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], 
                          "Spatial carbon": {"1": "330, 520, 560, 610", "2": "550, 340, 520, 600", "3": "590, 520, 340, 630", "4": "620, 560, 500, 330", "5": "360, 380, 560, 620"}},
                "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], 
                          "Spatial carbon": {"1": "510, 330, 550, 600", "2": "540, 500, 320, 610", "3": "310, 520, 550, 630", "4": "620, 540, 500, 340", "5": "320, 410, 560, 640"}},
                "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], 
                          "Spatial carbon": {"1": "540, 500, 320, 600", "2": "320, 510, 540, 600", "3": "560, 330, 520, 610", "4": "620, 560, 500, 330", "5": "330, 420, 550, 640"}},
                "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], 
                          "Spatial carbon": {"1": "320, 520, 560, 600", "2": "550, 330, 520, 580", "3": "600, 540, 500, 320", "4": "560, 500, 330, 540", "5": "500, 340, 560, 630"}},
                "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], 
                          "Spatial carbon": {"1": "510, 330, 560, 600", "2": "560, 500, 320, 590", "3": "320, 520, 540, 620", "4": "630, 560, 510, 340", "5": "330, 420, 560, 630"}},
                "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], 
                          "Spatial carbon": {"1": "540, 500, 320, 610", "2": "320, 510, 560, 620", "3": "560, 340, 520, 610", "4": "640, 560, 510, 330", "5": "520, 330, 540, 600"}},
                "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], 
                          "Spatial carbon": {"1": "330, 520, 560, 610", "2": "540, 330, 520, 600", "3": "580, 540, 330, 620", "4": "630, 560, 500, 330", "5": "520, 330, 550, 600"}}
            },
            "alpha": 40.00, "beta": 0.50, "gamma": 12.00
        }

    def _extract_agent_specific_data(self, scenario):
        """Extracts and formats relevant data structures for the agent's daily decisions."""
        
        day_keys = sorted([k for k in scenario['days'].keys() if k.startswith('Day')])
        self.daily_data = {}
        
        for i, day_key in enumerate(day_keys):
            day_info = scenario['days'][day_key]
            day_index = i + 1
            
            # Spatial Carbon for the agent's location (Location 5)
            spatial_str = day_info['Spatial carbon'].get(str(self.location))
            if spatial_str:
                sc_list = [float(x) for x in spatial_str.split(', ')]
            else:
                # Fallback if location data is missing for this day
                sc_list = scenario['carbon_intensity']

            self.daily_data[day_index] = {
                'tariff': day_info['Tariff'],
                'carbon': day_info['Carbon'],
                'baseline': day_info['Baseline load'],
                'spatial_carbon': sc_list,
            }
            
        # Update agent parameters from scenario if available
        self.alpha = scenario.get('alpha', self.alpha)
        self.beta = scenario.get('beta', self.beta)
        self.gamma = scenario.get('gamma', self.gamma)
        self.capacity = scenario.get('capacity', 7.0) # Default capacity if not specified

    def calculate_cost_function(self, day_index, slot_index, session_count):
        """
        Calculates the total score for choosing a specific slot, combining
        Cost (Price/Carbon), Congestion (Baseline adherence), and Comfort (Local).
        
        Cost function structure based on common DEX objectives:
        Score = alpha * (Weighted_Cost) + gamma * (Congestion_Penalty) + beta * (Comfort_Penalty)
        """
        
        data = self.daily_data[day_index]
        
        # --- 1. Cost Term (Minimizing Price/Carbon) ---
        # We combine price and carbon intensity into a single 'cost' metric.
        # Since carbon intensity values are generally higher than prices, we normalize them roughly.
        # Using Carbon Intensity as the primary driver for global goals (Carbon focus in Stage 3).
        
        # Get Global Carbon Intensity for the slot (from scenario header, used as noise approximation)
        global_carbon = self.scenario['carbon_intensity'][slot_index]
        
        # Use Day-Specific Carbon Intensity (Primary Global Weight)
        C_day = data['carbon'][slot_index]
        
        # Use Day-Specific Price
        P_day = data['tariff'][slot_index]
        
        # Agent 5 is a high-value (high demand) user, so carbon is important (Position 5 graduate tenant)
        # We prioritize carbon heavily here.
        weighted_cost = (self.alpha * C_day) + (self.alpha * 0.1 * P_day)
        
        # --- 2. Congestion Term (Capacity Adherence) ---
        # We must consider how our session count affects the capacity.
        
        # Baseline Load for the slot
        B_day = data['baseline'][slot_index]
        
        # Agent's base demand for this slot
        D_agent = self.base_demand[slot_index]
        
        # The congestion penalty is incurred if the resulting total load significantly exceeds the baseline,
        # or if the agent's contribution is high relative to baseline.
        # We assume the baseline load incorporates expected neighbor activity.
        
        # Penalty proportional to how much the agent's required load exceeds the available margin (Capacity - Baseline)
        # Since we don't know *other* agents' schedules, we use the baseline as the expected load floor.
        
        # We penalize large loads that push the system past expected baseline + capacity buffer.
        # Since session_count is binary (0 or 1 for this agent in this step), we use D_agent * session_count
        
        # Simplified congestion score: Penalize if demand is high relative to baseline.
        # Using gamma to penalize high demand relative to the baseline *if* the load is scheduled.
        congestion_penalty = self.gamma * (D_agent * session_count) / B_day
        
        # --- 3. Comfort Term (Local Penalty) ---
        # Agent 5 (Graduate tenant) prioritizes comfort/convenience (late commute implies flexibility post-22h, slot 3 is late).
        # But the general objective is *not* purely local optimization.
        
        # Agent 5 has a high base demand in slot 3 (0.90) and relatively low in slots 0/1.
        # This suggests late charging might be preferred/required by the user's profile.
        
        # Comfort penalty: Since we don't have explicit comfort mapping (like neighbor preferences),
        # we use neighbor example 4 (retirees) preference for early slots (0, 3) and neighbor 1 (engineer) for (0, 2).
        # Agent 5 is Position 5, so we look at their demand profile: high demand late (0.90 in slot 3).
        # We define a soft preference against slots that are generally less demanded by high-load users (slot 2=0.60).
        
        comfort_score = 0.0
        
        # Agent 5 profile suggests flexibility but maybe a slight bias away from slot 2 (0.60 base demand)
        # and towards slot 3 (0.90 base demand).
        if slot_index == 2:
            comfort_score = 1.0 # Mild penalty for slot 2
        elif slot_index == 3:
            comfort_score = 0.2 # Mild reward for slot 3
            
        local_comfort_penalty = self.beta * comfort_score
        
        # --- 4. Coordination Term (Neighbor Avoidance) ---
        # We must coordinate to avoid simultaneous high loading if neighbors are present in those slots.
        # Since we don't know the future state of neighbors, we use their known *preferences* as a heuristic for coordination.
        # We prefer slots where neighbors *don't* prefer, especially if neighbors have high local penalties (like N4 comfort=0.16).
        
        coordination_penalty = 0.0
        # If we choose slot 3, Neighbor 4 strongly prefers it. Coordination suggests avoiding it if possible.
        if slot_index == 3:
            # If N4 strongly prefers slot 3, and N1 dislikes it (prefers 0, 2), we slightly increase cost to avoid clashes with N4.
            coordination_penalty += 0.1 * self.beta # Small penalty to coordinate away from N4's favorite
        
        # If we choose slot 0, N1 prefers it.
        if slot_index == 0:
            coordination_penalty += 0.05 * self.beta # Very minor penalty to coordinate away from N1's favorite
            
        total_score = weighted_cost + congestion_penalty + local_comfort_penalty + coordination_penalty
        
        return total_score

    def run_optimization(self):
        """Determines the best slot for each of the 7 days."""
        
        recommendations = []
        day_keys = sorted([k for k in self.scenario['days'].keys() if k.startswith('Day')])
        
        for i, day_key in enumerate(day_keys):
            day_index = i + 1
            
            min_score = float('inf')
            best_slot = -1
            
            # Constraints check (Session counts)
            slot_limits = self.scenario['slot_min_sessions']
            slot_max = self.scenario['slot_max_sessions']

            # Iterate through all 4 slots (j = 0 to 3)
            for j in range(self.num_slots):
                
                # Determine session count. Since this is a static recommendation for the day,
                # and we don't have iterative refinement, we assume the agent will aim for 1 session
                # if the slot is viable (meets min/max session counts for the *collective*).
                # Since this agent is making ONE decision for the entire day, we assume session_count = 1 
                # if the slot is within min/max bounds (which are 1 or 2 for all slots).
                
                # For simplicity in this static 1-slot-per-day recommendation, we check feasibility
                # based on session limits: if slot j is even available (min sessions >= 1), we score it assuming one session.
                
                # If this agent must choose *one* slot for charging for the whole day, 
                # we check if scheduling 1 session of its demand fits the slot constraints.
                
                session_count = 1 # Assume one session of D_agent is being scheduled
                
                is_feasible = (slot_limits.get(str(j), 1) <= session_count <= slot_max.get(str(j), 2))
                
                if not is_feasible:
                    # Skip slots that are impossible based on static session constraints
                    continue
                    
                score = self.calculate_cost_function(day_index, j, session_count)
                
                if score < min_score:
                    min_score = score
                    best_slot = j
            
            if best_slot == -1:
                # Fallback: If no slot was scored (e.g., due to complex unhandled constraints), pick the lowest index.
                best_slot = 0
                
            recommendations.append(best_slot)
            
        return recommendations

    def save_output(self, recommendations):
        """Saves the final slot recommendations to global_policy_output.json."""
        
        output_data = {
            "agent_id": self.agent_id,
            "scenario_id": self.scenario.get('scenario_id', 'ev_peak_sharing_1'),
            "recommendations": [int(r) for r in recommendations]
        }
        
        # Output format requires a simple list of 7 indices.
        final_output = [int(r) for r in recommendations]

        with open('global_policy_output.json', 'w') as f:
            json.dump(final_output, f, indent=4)
        
        # For debugging/verification, we print the recommended slots
        print(f"Agent 5 Recommendations (7 days): {final_output}")


if __name__ == '__main__':
    policy = EVSchedulingPolicy(agent_id=5)
    policy.load_scenario()
    
    # Run optimization to get 7 daily slot recommendations
    daily_recommendations = policy.run_optimization()
    
    # Save results
    policy.save_output(daily_recommendations)

    # Ensure policy.py is executable and produces the required JSON file structure.
    # The structure relies on day order matching scenario order (Day 1 -> Index 0, Day 7 -> Index 6).