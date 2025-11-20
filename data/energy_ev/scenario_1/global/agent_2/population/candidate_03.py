import json
import numpy as np

class Policy:
    def __init__(self):
        # Load scenario data from scenario.json
        try:
            with open('scenario.json', 'r') as f:
                self.scenario = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("scenario.json not found. Ensure it is in the execution directory.")

        self.agent_id = "Agent 2"
        self.location_id = self.scenario['location']
        
        # Agent configuration
        self.alpha = self.scenario['alpha']
        self.beta = self.scenario['beta']
        self.gamma = self.scenario['gamma']
        
        self.slots_info = {i: {'time': self.scenario['slots'][i], 'price': self.scenario['price'][i], 
                                'carbon': self.scenario['carbon_intensity'][i]} 
                           for i in range(len(self.scenario['slots']))}
        
        self.num_slots = len(self.scenario['slots'])
        self.capacity = self.scenario['capacity']
        
        # Agent specific data
        self.base_demand = np.array(self.scenario['base_demand'])
        
        # Neighbor data processing
        self.neighbors = {}
        for neighbor in self.scenario['neighbor_examples']:
            neighbor_loc = neighbor['location']
            self.neighbors[neighbor_loc] = {
                'base_demand': np.array(neighbor['Base demand']),
                'comfort_penalty': neighbor['Comfort penalty'],
                'preferred_slots': neighbor['Preferred slots'],
                'ground_truth_min_cost_slots': neighbor['Ground truth min-cost slots by day']
            }
            
        # Day data processing
        self.days_data = {}
        self.day_names = list(self.scenario['days'].keys())
        for day_name, data in self.scenario['days'].items():
            # Extract day index (e.g., Day 1 -> 0, Day 2 -> 1)
            day_index = int(day_name.split(' ')[1]) - 1
            
            spatial_carbon_map = {}
            # Parse spatial carbon data for all locations (1 to 5)
            for loc_str, carbon_values in data['Spatial carbon'].items():
                spatial_carbon_map[int(loc_str)] = np.array([float(c) for c in carbon_values.split(', ')])
                
            self.days_data[day_index] = {
                'day_name': day_name,
                'tariff': np.array(data['Tariff']),
                'carbon': np.array(data['Carbon']),
                'baseline_load': np.array(data['Baseline load']),
                'spatial_carbon': spatial_carbon_map
            }

    def calculate_utility(self, day_idx, slot_idx, session_count):
        """
        Calculates the utility score for selecting a specific slot on a given day.
        Utility = - (Alpha * Carbon + Beta * Price + Gamma * Congestion + Comfort_Penalty)
        Since we maximize utility, we minimize the negative score.
        
        For Agent 2 (Feeder Analyst), the primary concern is transformer headroom (congestion).
        """
        day_data = self.days_data[day_idx]
        
        # --- 1. Global Metrics (Carbon and Price) ---
        # Carbon: Minimize instantaneous carbon intensity
        carbon_cost = day_data['carbon'][slot_idx]
        
        # Price: Minimize tariff
        price_cost = day_data['tariff'][slot_idx]
        
        # --- 2. Local Congestion/Headroom Metric (Spatial Carbon) ---
        # Agent 2 is in location 2. The problem description emphasizes transformer headroom.
        # For feeder analysts, local spatial carbon is a proxy for congestion stress on local assets.
        spatial_carbon_cost = day_data['spatial_carbon'][self.location_id][slot_idx]

        # --- 3. Baseline Load (Proxy for overall feeder stress) ---
        # Use baseline load as a component of congestion if spatial carbon is already used.
        # Baseline load itself might be related to the collective state.
        # Given the focus is transformer headroom (local), we weigh spatial carbon heavily.
        
        # --- 4. Comfort/Session Count Penalty ---
        # Comfort penalty is usually associated with deviation from the baseline/preferred schedule.
        # For Agent 2, we assume comfort is minimally penalized unless forced by constraints.
        # We use the session count (which determines how much load is added) to scale a small penalty.
        comfort_penalty = session_count * 0.01 # Minor penalty for usage level
        
        # --- 5. Neighbor Coordination (Implicit) ---
        # Coordination is achieved by observing neighbor behavior and trying to avoid synchronous peaks.
        # Since this decision is made slot-by-slot *before* neighbors choose, we use neighbor *preferences*
        # as a signal for potential conflict zones (i.e., slots they strongly prefer are high conflict potential).
        
        conflict_signal = 0
        for loc, data in self.neighbors.items():
            if slot_idx in data['preferred_slots']:
                # Penalize slots heavily preferred by neighbors, especially if they are high congestion slots globally.
                conflict_signal += data['comfort_penalty'] * 1.5 
        
        # --- Objective Function Formulation (Minimization) ---
        # We want to minimize: alpha*Carbon + beta*Price + gamma*Congestion + Coordination_Penalty
        
        # Congestion Metric: Prioritize spatial carbon (local headroom) heavily, as per persona.
        # Since Agent 2 prioritizes headroom (congestion), gamma weights the congestion term significantly.
        
        # Heuristic Weights Combination:
        # Agent 2 prioritizes Congestion (Headroom) -> High Gamma weight on spatial_carbon_cost
        # Agent 2 minimizes Carbon (Global) -> Medium Alpha weight
        # Agent 2 minimizes Price (Implied/General good practice) -> Low Beta weight
        
        # Let's define costs:
        cost = (self.alpha * carbon_cost) + \
               (self.beta * price_cost) + \
               (self.gamma * spatial_carbon_cost) + \
               conflict_signal + \
               comfort_penalty
               
        # The utility score is the negative of the cost for maximization algorithms, 
        # but here we just minimize the composite cost.
        return cost

    def get_session_recommendation(self, day_idx):
        day_name = self.day_names[day_idx]
        day_constraints = self.days_data[day_idx]
        
        min_sessions = self.scenario['slot_min_sessions'][str(slot_idx)]
        max_sessions = self.scenario['slot_max_sessions'][str(slot_idx)]
        
        # Determine session count for this day based on constraints and known global signals (e.g., maintenance)
        # For Stage 3 (Collective), we must choose a single session count per slot for the day, 
        # which implies we need to decide on the allocation level based on the scenario context.
        
        # Contextual Adjustments based on Day Description:
        day_description = self.scenario['days'][day_name]['description']
        
        # Default assumption: Use baseline max capacity if not constrained.
        # Since this is Agent 2 (EV), base demand is up to 0.7 kW per slot.
        
        # We need to select a single session configuration (e.g., 1 or 2 sessions) for each slot
        # that minimizes the overall cost for the day, respecting min/max bounds.
        
        daily_session_counts = {}
        
        # Iteratively search for the best session count (1 or 2, based on min/max) for each slot
        slot_costs = {}
        
        for slot_idx in range(self.num_slots):
            min_s = self.scenario['slot_min_sessions'][str(slot_idx)]
            max_s = self.scenario['slot_max_sessions'][str(slot_idx)]
            
            best_cost = float('inf')
            best_s = -1
            
            # Try possible session counts {min_s, ..., max_s}
            # Since min/max are usually 1 or 2, we test 1 and 2 if they are within bounds.
            possible_sessions = sorted(list(set([min_s, max_s] + [1, 2])))
            possible_sessions = [s for s in possible_sessions if min_s <= s <= max_s]
            
            if not possible_sessions: # Should not happen if constraints are sensible
                possible_sessions = [min_s] 

            for s in possible_sessions:
                # Calculate utility assuming 's' sessions are running
                cost = self.calculate_utility(day_idx, slot_idx, session_count=s)
                
                # Apply Day-Specific Constraint Handling (e.g., rationing)
                if "rationed" in day_description and slot_idx == 2:
                    # Day 6: Maintenance advisory caps slot 2
                    if s > 1: cost = float('inf') # Force session count to 1 if rationed
                
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            
            slot_costs[slot_idx] = (best_cost, best_s)
            daily_session_counts[slot_idx] = best_s

        # --- 6. Collective Coordination (Transformer Capacity Check) ---
        # Agent 2 is a feeder analyst, primarily concerned with capacity (Transformer Headroom).
        # Capacity constraint: Sum of loads <= 6.8 MW.
        # We must ensure the chosen sessions do not violate the capacity limit, 
        # potentially adjusting downwards if the sum of minimum sessions exceeds capacity 
        # or if preferred choices lead to overload.
        
        # Estimate Total Load based on chosen sessions and baseline load
        # Load contribution = session_count * (Base_Demand_for_Agent_2 + neighbor_demand_proxy)
        # This is complex as we only know *our* base demand and *neighbor preferences*.
        
        # Simplified capacity check: Assume a fixed load factor for the session count.
        # Let's assume 1 session ≈ 0.5 kW load contribution for simplicity in coordination modeling, 
        # relative to the 6.8 MW capacity.
        
        SESSION_LOAD_UNIT = 0.5 # kW per session chosen by an agent/neighbor type
        
        # Calculate predicted total load based on chosen sessions (Agent 2 only)
        agent_load = sum(daily_session_counts[s] * SESSION_LOAD_UNIT for s in range(self.num_slots))
        
        # Estimate Neighbor Load Contribution (Simplification: Assume neighbors choose their preferred slots)
        neighbor_load = 0
        for loc, data in self.neighbors.items():
            # Check if neighbors have a known configuration for this day (they don't explicitly here, use ground truth)
            gt_slot = data['ground_truth_min_cost_slots'][day_idx]
            # Assume neighbor uses 1 session if their preferred slot is chosen.
            neighbor_load += 1 * SESSION_LOAD_UNIT 

        total_predicted_load = agent_load + neighbor_load
        
        # If we are heavily loaded, prioritize lower session counts, especially in high-gamma slots.
        if total_predicted_load > self.capacity * 0.95: # Over 95% capacity threshold
            # Re-evaluate slots, penalizing session counts > 1 heavily if capacity is near limit.
            
            reassigned_counts = daily_session_counts.copy()
            
            # Iterate slots in order of increasing cost (lowest cost slot first) to see if we must drop load
            sorted_slots = sorted(slot_costs.keys(), key=lambda k: slot_costs[k][0])
            
            current_load = total_predicted_load
            
            for slot_idx in sorted_slots:
                if current_load <= self.capacity * 0.95:
                    break
                
                current_s = reassigned_counts[slot_idx]
                
                # Can we reduce sessions in this slot? (Must stay >= min_s)
                min_s_slot = self.scenario['slot_min_sessions'][str(slot_idx)]
                
                if current_s > min_s_slot:
                    # Drop one session
                    reassigned_counts[slot_idx] -= 1
                    current_load -= SESSION_LOAD_UNIT
        else:
            reassigned_counts = daily_session_counts


        # --- Final Selection: Choose the slot that yields the minimum cost based on the determined session count ---
        
        final_slot_recommendation = -1
        min_final_cost = float('inf')
        
        for slot_idx in range(self.num_slots):
            s = reassigned_counts[slot_idx]
            cost = self.calculate_utility(day_idx, slot_idx, session_count=s)
            
            # Critical check: If the slot is still over-capacity due to fixed neighbor choices, 
            # we must pick the best available slot that meets its minimum session requirement.
            
            if cost < min_final_cost:
                min_final_cost = cost
                final_slot_recommendation = slot_idx
        
        # Ensure the selected slot meets its minimum session requirement (1 in this scenario)
        if reassigned_counts[final_slot_recommendation] < 1:
            # Fallback: If the minimum cost slot requires 0 sessions due to capacity crunch 
            # (which shouldn't happen if min_s=1), pick the lowest cost slot that allows 1 session.
            best_fallback = -1
            min_fallback_cost = float('inf')
            for slot_idx in range(self.num_slots):
                if reassigned_counts[slot_idx] >= 1:
                    cost = self.calculate_utility(day_idx, slot_idx, session_count=1)
                    if cost < min_fallback_cost:
                        min_fallback_cost = cost
                        best_fallback = slot_idx
            final_slot_recommendation = best_fallback


        return final_slot_recommendation

    def generate_policy(self):
        recommendations = []
        
        # Iterate over the 7 days
        for day_idx in range(7):
            recommended_slot = self.get_session_recommendation(day_idx)
            recommendations.append(recommended_slot)
            
        return recommendations

# --- Execution ---
policy_engine = Policy()
slot_recommendations = policy_engine.generate_policy()

# Output formatting
output_data = {
    "Agent": policy_engine.agent_id,
    "Location": policy_engine.location_id,
    "Recommendations (7 Days)": slot_recommendations
}

# Save global_policy_output.json
with open('global_policy_output.json', 'w') as f:
    json.dump({"Agent 2 Recommendations": slot_recommendations}, f, indent=4)

# Save policy.py content (self-contained script)
# Since the output must *be* policy.py, we ensure the script executes fully when run.
# The execution part is already written above.
# The final required output is just the source code of this script.

# If running this script directly, it will generate policy.py and global_policy_output.json

# For the final required output, we just need the source code itself.
# We simulate the execution by ensuring the main logic runs when the file is executed.
if __name__ == '__main__':
    # Running the policy generation logic again to ensure file creation upon execution
    policy_engine = Policy()
    slot_recommendations = policy_engine.generate_policy()
    with open('global_policy_output.json', 'w') as f:
        json.dump({"Agent 2 Recommendations": slot_recommendations}, f, indent=4)

# The structure ensures that running `python policy.py` performs steps 1-4.
# The reasoning follows the Agent 2 persona: Prioritize local headroom (Spatial Carbon, high Gamma weight) 
# while mitigating known coordination hot-spots (neighbor preferred slots) and observing specific day constraints.
# We force session selection optimization before the final slot choice to satisfy the "collective" aspect.
# Given the 7-day lookahead, this heuristic is based purely on forecast data.