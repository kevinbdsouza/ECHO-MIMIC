import json
import numpy as np
import os

class Policy:
    def __init__(self):
        # Load scenario data from scenario.json, assuming it's in the same directory
        try:
            with open('scenario.json', 'r') as f:
                self.scenario = json.load(f)
        except FileNotFoundError:
            print("Error: scenario.json not found. Ensure the file is in the execution directory.")
            self.scenario = None
            return

        self.agent_id = "Agent 1" # Implicit from context, but useful for reference
        self.agent_loc = self.scenario['location']
        self.slots = self.scenario['slots']
        self.slot_keys = list(self.slots.keys())
        self.num_slots = len(self.slot_keys)
        self.num_days = 7

        # Agent specific data
        self.base_demand = np.array(self.scenario['base_demand'])

        # Global parameters
        self.alpha = self.scenario['alpha']
        self.beta = self.scenario['beta']
        self.gamma = self.scenario['gamma']

        # Neighbor data
        self.neighbors = self.scenario.get('neighbor_examples', [])
        self.neighbor_data = self._process_neighbor_data()
        
        # Capacity constraints (Total capacity for Agent 1)
        self.capacity = self.scenario['capacity']

        # Slot constraints (Min/Max sessions, implicitly relating to usage limits)
        self.slot_min_sessions = np.array([self.scenario['slot_min_sessions'][k] for k in self.slot_keys])
        self.slot_max_sessions = np.array([self.scenario['slot_max_sessions'][k] for k in self.slot_keys])
        
        # Daily constraints (Tariff, Carbon, Baseline, Spatial Carbon)
        self.daily_scenarios = {}
        for day_name, data in self.scenario['days'].items():
            day_index = int(day_name.split('Day ')[1].split(' ')[0]) - 1
            self.daily_scenarios[day_index] = {
                'tariff': np.array(data['Tariff']),
                'carbon': np.array(data['Carbon']),
                'baseline': np.array(data['Baseline load']),
                'spatial_carbon': self._process_spatial_carbon(data['Spatial carbon'])
            }

    def _process_spatial_carbon(self, sc_dict):
        """Processes spatial carbon dictionary into a structured numpy array."""
        # Assuming spatial carbon is provided for locations 1 through 5
        num_locs = len(self.neighbors) + 1 # Agent 1 + 5 neighbors (if present)
        
        # Initialize array: (Day Slot Index, Location Index) -> Carbon Intensity
        # Note: Location indices in the scenario input seem 1-based. We use 0-based internally.
        spat_array = np.zeros((self.num_slots, num_locs))
        
        for loc_str, values in sc_dict.items():
            loc_idx = int(loc_str) - 1 # Convert 1-based location ID to 0-based index
            if 0 <= loc_idx < num_locs:
                spat_array[:, loc_idx] = np.array(values)
        return spat_array

    def _process_neighbor_data(self):
        """Compiles neighbor data, prioritizing location-specific patterns."""
        data = {}
        for neighbor in self.neighbors:
            loc = neighbor['location']
            data[loc] = {
                'base_demand': np.array(neighbor['Base demand']),
                'preferred_slots': neighbor['Preferred slots'],
                'comfort_penalty': neighbor['Comfort penalty'],
                'ground_truth_usage': neighbor['ground_truth usage by day']
            }
        return data

    def calculate_costs(self, day_idx, usage_vector):
        """
        Calculates the total objective function value (Cost) for a given usage vector on a specific day.
        Cost = alpha * Total_Cost + beta * Total_Carbon + gamma * Total_Spatial_Carbon + Comfort_Penalty
        """
        
        day_data = self.daily_scenarios[day_idx]
        
        # 1. Financial Cost (Price * Usage * Baseline)
        # We must infer the actual power drawn/supplied. Assuming usage_vector * Capacity is the total energy shifted relative to some implicit baseline energy unit.
        # A simpler interpretation for demonstration is that the usage vector directly scales the costs provided.
        # We use the provided tariffs for the main schedule (global scope)
        tariffs = np.array(self.scenario['price'])
        
        # For daily cost calculation, we must use the specific day's tariff
        total_price_cost = np.sum(day_data['tariff'] * usage_vector * self.base_demand)
        
        # 2. Carbon Cost (Local Carbon Intensity * Usage * Baseline)
        total_carbon_cost = np.sum(day_data['carbon'] * usage_vector * self.base_demand)

        # 3. Spatial Carbon Cost (Coordination metric based on local feeder impact)
        # Agent 1 is at location 1 (index 0 in spatial carbon arrays)
        # Since this is Collective Stage 3, we focus on minimizing *overall* impact, 
        # which often means looking at the worst-affected areas OR coordinating based on observed neighbors.
        
        # For Agent 1 (Location 1, index 0), we use its own spatial carbon profile (index 0)
        # A proxy for coordination contribution: Minimize usage when *neighboring* areas (other locations) are high carbon.
        
        # Since Agent 1 is the focus, let's use its own location's spatial profile relative to its baseline load
        spatial_carbon_impact = np.sum(day_data['spatial_carbon'][:, 0] * usage_vector * self.base_demand)
        
        # 4. Comfort/Constraint Penalty (Usage deviation from neighbors/min/max)
        comfort_penalty = 0
        
        # a) Neighbor Alignment (Avoid conflicting with observed neighbor patterns heavily)
        # For this agent, we look at neighbors' ground truth usage relative to our base demand.
        # If neighbors heavily use slots where we are planning high usage, we incur a penalty, 
        # unless the context suggests we should join them (e.g., low price).
        
        # Simple coordination heuristic: If neighbors prefer specific slots, we penalize deviating significantly IF 
        # the global objective doesn't strongly pull us away.
        
        # For Agent 1, the global goal suggests prioritizing low carbon/price. 
        # We'll use neighbor preferences as a soft constraint on high usage.
        
        neighbor_penalty = 0
        for loc, data in self.neighbor_data.items():
            # Check if our usage is high where Neighbor loc *prefers* (high usage)
            # This is tricky in Stage 3 Collective where we only see neighbors' *past* usage.
            
            # Heuristic: If a neighbor shows strong preference (e.g., > 0.7 usage in preferred slots), 
            # and we use that slot heavily, we gain/lose based on whether that aligns with global goals.
            
            # Simpler: Penalize heavily if we violate min/max sessions, which are often linked to physical constraints.
            pass # We handle min/max bounds explicitly later.

        # Weighting the objective function components
        # Using the provided alpha, beta, gamma for the main trade-offs. Comfort is implicitly handled by constraints
        # and the coordination aspect via spatial carbon.
        
        total_cost = (self.alpha * total_price_cost + 
                      self.beta * total_carbon_cost + 
                      self.gamma * spatial_carbon_impact)
        
        return total_cost

    def calculate_comfort_and_constraints(self, usage_vector, day_idx):
        """Calculates penalties for comfort (deviation from base demand) and constraint violations."""
        penalty = 0.0
        
        # Constraint 1: Slot Min/Max Sessions (Usage must correspond to valid session count)
        # Assuming usage_vector represents the *fraction* of a session allocated [0, 1], 
        # and the min/max sessions relate to the total energy exchanged. 
        # In the absence of a clear session energy mapping, we treat usage_vector [0, 1] as the primary variable,
        # constrained by the implicit session bounds (e.g., if usage is 1.0, it counts as 1 session).
        
        # Heuristic: If usage exceeds the implied maximum session count (i.e., usage > slot_max_sessions[s] / MAX_SESSION_ENERGY), 
        # or falls below min (usage < slot_min_sessions[s] / MAX_SESSION_ENERGY), apply penalty.
        # Since we don't have MAX_SESSION_ENERGY, we enforce usage bounds related to the min/max slots themselves.
        
        # Use min/max usage constraints derived from session limits as hard bounds on the usage vector [0, 1] 
        # if the session limits imply strict participation levels.
        
        # A common setup is that max session maps to max usage fraction (e.g., 1.0), and min session maps to min usage fraction (e.g., 0.2).
        
        # Given the ambiguity, we enforce soft limits based on the ratio of min/max sessions relative to the maximum possible slots (2).
        
        # Let's use the provided min/max sessions as multipliers on a base unit of usage (e.g., 1.0 capacity).
        # Since we must return [0, 1], we interpret min/max sessions as strict operational boundaries on participation.
        
        # If min_session is 1 and max_session is 2 (out of 2 slots): usage must be >= 0.5 (1/2) and <= 1.0 (2/2).
        # If min_session is 1 and max_session is 1: usage must be exactly 0.5.
        
        # We define REQUIRED_USAGE based on the ratio to the maximum possible sessions (2)
        MAX_POSSIBLE_SESSIONS = 2.0
        
        required_min_usage = self.slot_min_sessions / MAX_POSSIBLE_SESSIONS
        required_max_usage = self.slot_max_sessions / MAX_POSSIBLE_SESSIONS
        
        # Ensure these are clamped within [0, 1]
        required_min_usage = np.clip(required_min_usage, 0.0, 1.0)
        required_max_usage = np.clip(required_max_usage, 0.0, 1.0)
        
        # Penalty for violating min usage requirement
        under_usage = np.maximum(0, required_min_usage - usage_vector)
        penalty += np.sum(under_usage) * self.gamma * 10 # High penalty for missing required participation

        # Penalty for exceeding max usage requirement
        over_usage = np.maximum(0, usage_vector - required_max_usage)
        penalty += np.sum(over_usage) * self.gamma * 10 

        # Constraint 2: Total Capacity Limit (Must not exceed capacity C=6.8, implicitly scaled by baseline/time)
        # Since usage is normalized [0, 1], we must scale it by some inferred maximum potential draw/feed.
        # If we assume total capacity 6.8 MW is the hard limit for the whole day, or average over the day:
        # Total energy = sum(usage_vector * base_demand) must be < Capacity * 4 (4 slots).
        
        # However, the capacity constraint usually applies to instantaneous power or a local transformer rating.
        # Given Agent 1 is Location 1, we look at spatial carbon for Day X, Location 1: day_data['spatial_carbon'][:, 0]
        
        # The constraint given is capacity=6.8. This is likely the total energy/charge capacity for the agent over the defined period.
        # Let's assume the capacity C is the *maximum total usage integral* for the 4 slots.
        
        TOTAL_CAPACITY_PENALTY_FACTOR = 50.0
        
        # We assume the total required energy integration must be respected, perhaps relative to a baseline sum.
        # Given the context of EV charging, this usually implies a total charge limit.
        # Since usage is relative [0, 1], we use the sum of usage scaled by baseline load as a proxy for total energy.
        
        # We check if the sum of weighted usage exceeds a threshold derived from capacity. 
        # For simplicity, we check the sum of usage * demand relative to a scaled capacity.
        # Since we don't know the exact scaling, we use the sum of usage as the primary constraint factor for penalty.
        
        SUM_USAGE = np.sum(usage_vector)
        # If capacity C=6.8 suggests a hard limit on participation, we set a soft penalty threshold based on slot count.
        # A typical usage might be around 4 slots * 1.0 capacity unit = 4.0. Exceeding 4.0 should be penalized if C is interpreted as max total normalized usage.
        
        # If C=6.8 is the hard limit on *energy*, we must use the baseline load as the energy unit.
        # Total Base Load Sum = 5.2+5.0+4.9+6.5 = 21.6 (Day 0 Global)
        # If we use the capacity C=6.8 as the maximum *average* capacity utilization scaled by time:
        
        # Heuristic: Penalize usage sum significantly if it exceeds 3.5 (a plausible normalized total limit over 4 slots).
        if SUM_USAGE > 3.5:
            penalty += (SUM_USAGE - 3.5) * TOTAL_CAPACITY_PENALTY_FACTOR
            
        # Comfort Penalty (If we had agent-specific comfort preferences)
        # Since we only have neighbor comfort penalties, we skip explicit comfort calculation for Agent 1 here,
        # relying instead on price/carbon minimization and explicit constraint adherence.
        
        return penalty

    def generate_usage_vector(self, day_idx):
        """
        Generates the usage vector for a given day using an optimization approach
        (simulated by iterating over possible load shapes and selecting the best one).
        """
        day_data = self.daily_scenarios[day_idx]
        
        # 1. Determine the best cost profile based on global factors (Price/Carbon/Spatial)
        
        # Since we cannot use iterative optimization solvers (like CVXPY), we generate candidate vectors
        # based on heuristic priorities: Low Carbon, Low Price, respecting neighbor coordination.
        
        # Candidate generation based on prioritizing the cheapest/greenest slots globally:
        
        # Combine Price and Carbon for daily preference ranking (Lower is better)
        # Using Global Price/Carbon for ranking candidates, as daily specific data is used in cost evaluation
        global_price = np.array(self.scenario['price'])
        global_carbon = np.array(self.scenario['carbon_intensity'])
        
        # Weighting Factor: Prioritize Carbon (beta=0.5) over Price (alpha=40.0) if costs were equal, but alpha is huge.
        # Since alpha (40) is much larger than beta (0.5), Price dominates, unless carbon is extremely high.
        # Given the difference, we rely on the cost function evaluation below.
        
        # Candidate 1: Use the lowest price/carbon slots heavily (Price-driven base)
        
        # Create a ranking score for each slot based on global factors for initial prioritization
        ranking_score = self.alpha * global_price + self.beta * global_carbon
        
        # Create a set of candidate usage profiles (U_c)
        candidates = []
        
        # Candidate structure: [0, 0, 0, 0] to [1, 1, 1, 1] -- We need smarter candidates.
        
        # Candidate 1: Maximize usage in the best ranked slot (0, 1, 2, or 3)
        for best_slot in range(self.num_slots):
            candidate_u = np.zeros(self.num_slots)
            # If slot 'best_slot' is best, try to use it up to its max session capacity
            candidate_u[best_slot] = required_max_usage[best_slot] 
            
            # Distribute remaining energy to other slots based on their ranking score
            
            # A simple approach: Find the absolute best slot (lowest ranking_score)
            # and try to load it up, while ensuring minimum required usage elsewhere.
            
            
        # ---- Simplified Greedy Approach based on Daily Cost Evaluation ----
        
        # 1. Calculate baseline usage based on minimum requirements
        initial_usage = required_min_usage.copy()
        
        # 2. Calculate remaining flexibility (Capacity to shift/add usage)
        # Total available energy to distribute: Sum(Max Usage) - Sum(Min Usage)
        total_flexibility = np.sum(required_max_usage - required_min_usage)
        
        # 3. Rank slots based on the *Daily* objective cost gradient (how much cost decreases by increasing usage there)
        # Since we cannot simulate gradient descent, we rank based on the inverse of the daily cost factors.
        
        # Inverse Daily Cost Ranking (Lower value means better slot to add energy to)
        # Note: Since usage is multiplicative with the cost factors, lower cost factor means lower instantaneous cost.
        daily_cost_factors = (day_data['tariff'] * self.base_demand * self.alpha +
                              day_data['carbon'] * self.base_demand * self.beta +
                              day_data['spatial_carbon'][:, 0] * self.base_demand * self.gamma)
        
        # We want to prioritize slots where cost_factors are minimal.
        
        # Create a list of (cost_factor, index) tuples and sort ascendingly
        sorted_slots = sorted([(daily_cost_factors[i], i) for i in range(self.num_slots)])
        
        final_usage = initial_usage.copy()
        
        # Distribute the total flexibility capacity greedily among the best slots, 
        # respecting their individual max usage bounds.
        
        # We iterate through slots in order of best daily cost factor
        for cost, slot_idx in sorted_slots:
            # How much more can we add to this slot?
            can_add = required_max_usage[slot_idx] - final_usage[slot_idx]
            
            if can_add > 1e-6:
                # Since we are distributing 'flexibility' derived from max_session limits, 
                # we add up to the remaining capacity for that slot.
                final_usage[slot_idx] += can_add
                final_usage[slot_idx] = np.clip(final_usage[slot_idx], 0, 1.0)
                
        # 4. Post-processing: Ensure total usage respects capacity (if applicable, handled by constraints in cost calc)
        # Since the greedy approach respected individual max usages derived from session limits, 
        # we rely on the constraint check in the cost function to penalize capacity violations.
        
        # 5. Final Constraint Check (Re-apply hard safety clips if any theoretical calculation allowed overshoot)
        final_usage = np.clip(final_usage, required_min_usage, required_max_usage)

        # The final usage vector is the result of prioritizing minimum required usage, then filling based on daily cost minimization.
        return final_usage

    def run_policy(self):
        if not self.scenario:
            return

        all_day_usages = []
        
        # Pre-calculate constraints based on day-independent data (Min/Max sessions)
        MAX_POSSIBLE_SESSIONS = 2.0
        global required_max_usage
        global required_min_usage
        required_min_usage = np.array([self.slot_min_sessions[k] for k in self.slot_keys]) / MAX_POSSIBLE_SESSIONS
        required_max_usage = np.array([self.slot_max_sessions[k] for k in self.slot_keys]) / MAX_POSSIBLE_SESSIONS
        required_min_usage = np.clip(required_min_usage, 0.0, 1.0)
        required_max_usage = np.clip(required_max_usage, 0.0, 1.0)
        
        
        for day_idx in range(self.num_days):
            usage_vector = self.generate_usage_vector(day_idx)
            
            # Verification step (Optional: Check final cost)
            # cost = self.calculate_costs(day_idx, usage_vector)
            # penalty = self.calculate_comfort_and_constraints(usage_vector, day_idx)
            
            # Ensure final output is cleanly formatted as a list of floats [0.0, 1.0]
            all_day_usages.append(usage_vector.tolist())
            
        # Output generation
        output_data = {
            "Agent 1 Usage Profile": {
                "Day 1": all_day_usages[0],
                "Day 2": all_day_usages[1],
                "Day 3": all_day_usages[2],
                "Day 4": all_day_usages[3],
                "Day 5": all_day_usages[4],
                "Day 6": all_day_usages[5],
                "Day 7": all_day_usages[6],
            }
        }

        # Save output to global_policy_output.json
        with open('global_policy_output.json', 'w') as f:
            json.dump(all_day_usages, f, indent=4)
            
        # Since the required output format is just the list of 7 vectors, we save that list.
        # If the prompt implicitly asks for a structure matching the neighbor examples, 
        # we stick to the explicit instruction: "Write global_policy_output.json containing a list of seven usage vectors"

# Execution
if __name__ == '__main__':
    # Mock scenario.json loading environment if running outside the simulation directory structure
    # This part is usually handled by the execution environment, but needed here for completeness if run stand-alone.
    
    # If scenario.json is not present, create a minimal mock file for testing the logic structure
    if not os.path.exists('scenario.json'):
        print("Mocking scenario.json for execution test...")
        mock_scenario = {
            "scenario_id": "ev_peak_sharing_1",
            "slots": {k: v for k, v in zip(["0", "1", "2", "3"], ["19-20", "20-21", "21-22", "22-23"])},
            "price": [0.23, 0.24, 0.27, 0.30],
            "carbon_intensity": [700, 480, 500, 750],
            "capacity": 6.8,
            "baseline_load": [5.2, 5.0, 4.9, 6.5],
            "slot_min_sessions": {"0": 1, "1": 1, "2": 1, "3": 1},
            "slot_max_sessions": {"0": 2, "1": 2, "2": 1, "3": 2},
            "days": {
                "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": {
                    "1": [330, 520, 560, 610], "2": [550, 340, 520, 600], "3": [590, 520, 340, 630], "4": [620, 560, 500, 330], "5": [360, 380, 560, 620]
                }},
                 "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": {
                    "1": [510, 330, 550, 600], "2": [540, 500, 320, 610], "3": [310, 520, 550, 630], "4": [620, 540, 500, 340], "5": [320, 410, 560, 640]
                }},
                "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": {
                    "1": [540, 500, 320, 600], "2": [320, 510, 540, 600], "3": [560, 330, 520, 610], "4": [620, 560, 500, 330], "5": [330, 420, 550, 640]
                }},
                "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": {
                    "1": [320, 520, 560, 600], "2": [550, 330, 520, 580], "3": [600, 540, 500, 320], "4": [560, 500, 330, 540], "5": [500, 340, 560, 630]
                }},
                "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": {
                    "1": [510, 330, 560, 600], "2": [560, 500, 320, 590], "3": [320, 520, 540, 620], "4": [630, 560, 510, 340], "5": [330, 420, 560, 630]
                }},
                "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": {
                    "1": [540, 500, 320, 610], "2": [320, 510, 560, 620], "3": [560, 340, 520, 610], "4": [640, 560, 510, 330], "5": [520, 330, 540, 600]
                }},
                "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": {
                    "1": [330, 520, 560, 610], "2": [540, 330, 520, 600], "3": [580, 540, 330, 620], "4": [630, 560, 500, 330], "5": [520, 330, 550, 600]
                }}
            },
            "alpha": 40.00, "beta": 0.50, "gamma": 12.00,
            "location": 1,
            "base_demand": [1.20, 0.70, 0.80, 0.60],
            "neighbor_examples": [
                {"location": 2, "Base demand": [0.70, 1.00, 0.80, 0.50], "Preferred slots": [1, 2], "Comfort penalty": 0.14, "ground_truth usage by day": {"Day 1": [0.08, 0.77, 0.09, 0.06], "Day 2": [0.08, 0.12, 0.74, 0.06], "Day 3": [0.73, 0.12, 0.09, 0.06], "Day 4": [0.08, 0.77, 0.09, 0.06], "Day 5": [0.08, 0.12, 0.74, 0.06], "Day 6": [0.73, 0.12, 0.09, 0.06], "Day 7": [0.08, 0.77, 0.09, 0.06]}},
                {"location": 3, "Base demand": [0.60, 0.80, 0.90, 0.70], "Preferred slots": [1, 3], "Comfort penalty": 0.20, "ground_truth usage by day": {"Day 1": [0.07, 0.09, 0.75, 0.08], "Day 2": [0.72, 0.09, 0.10, 0.08], "Day 3": [0.07, 0.74, 0.10, 0.08], "Day 4": [0.07, 0.09, 0.10, 0.73], "Day 5": [0.72, 0.09, 0.10, 0.08], "Day 6": [0.07, 0.74, 0.10, 0.08], "Day 7": [0.07, 0.09, 0.75, 0.08]}}
            ]
        }
        with open('scenario.json', 'w') as f:
            json.dump(mock_scenario, f, indent=4)

    policy = Policy()
    policy.run_policy()
    print("Policy executed. Usage vectors saved to global_policy_output.json")