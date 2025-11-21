import json
import numpy as np
import os

class DistributedEnergyPolicy:
    def __init__(self):
        # Define file paths relative to the execution context
        self.scenario_file = 'scenario.json'
        
        # Agent specific parameters (Position 1, Battery Engineer)
        self.location_id = 1
        self.base_demand = np.array([1.20, 0.70, 0.80, 0.60])
        self.alpha = 40.00  # Cost sensitivity (Carbon/Price)
        self.beta = 0.50    # Coordination factor
        self.gamma = 12.00  # Comfort sensitivity
        
        # Global constraints/context from scenario file (will be loaded)
        self.scenario_data = None
        self.day_keys = None
        
        # Neighbor observed behavior and constraints
        self.neighbor_info = {
            2: {'location': 2, 'base_demand': np.array([0.70, 1.00, 0.80, 0.50]), 'preferred_slots': [1, 2], 'comfort_penalty': 0.14},
            3: {'location': 3, 'base_demand': np.array([0.60, 0.80, 0.90, 0.70]), 'preferred_slots': [1, 3], 'comfort_penalty': 0.20},
        }
        
        # Derived coordination constraints (simplified congestion/load balancing)
        # Since we are Agent 1 (Location 1), we primarily look at spatial carbon feedback
        # The neighbor examples show what they *prefer* to do, which suggests an implicit goal.

    def load_scenario(self):
        """Loads scenario.json from the current directory."""
        try:
            with open(self.scenario_file, 'r') as f:
                self.scenario_data = json.load(f)
            
            # Extract day keys based on the order they appear (Day 1 to Day 7)
            self.day_keys = [key for key in self.scenario_data['days'].keys()]
            
            # Load global context (which applies if day-specific context is not used)
            self.global_params = {
                'price': np.array(self.scenario_data['price']),
                'carbon_intensity': np.array(self.scenario_data['carbon_intensity']),
                'baseline_load': np.array(self.scenario_data['baseline_load']),
                'slot_min_sessions': self.scenario_data['slot_min_sessions'],
                'slot_max_sessions': self.scenario_data['slot_max_sessions'],
                'capacity': self.scenario_data['capacity']
            }

        except FileNotFoundError:
            print(f"Error: {self.scenario_file} not found.")
            exit(1)

    def calculate_daily_costs(self, day_key):
        """Calculates composite cost vector for a specific day."""
        day_data = self.scenario_data['days'][day_key]
        
        # Use day-specific data if available, otherwise fall back to global
        price = np.array(day_data.get('Tariff', self.global_params['price']))
        carbon = np.array(day_data.get('Carbon', self.global_params['carbon_intensity']))
        baseline = np.array(day_data.get('Baseline load', self.global_params['baseline_load']))
        
        # Location 1 Spatial Carbon Intensity (Agent's local congestion feedback)
        spatial_carbon_str = day_data.get('Spatial carbon', {}).get(str(self.location_id))
        if not spatial_carbon_str:
             # Fallback to global spatial carbon if location-specific data is missing (shouldn't happen based on prompt structure)
             spatial_carbon = self.global_params['carbon_intensity'] 
        else:
            spatial_carbon = np.array([float(x) for x in spatial_carbon_str.split('; ')])

        # --- Objective Function Components ---
        
        # 1. Carbon Cost (Weighted by Alpha, incorporating local environmental factors)
        # We use the agent's specific spatial carbon as a proxy for immediate local stress/environmental goal alignment.
        carbon_cost = self.alpha * carbon * spatial_carbon 
        
        # 2. Price Cost (Standard economic consideration)
        price_cost = price
        
        # 3. Comfort Cost (Inverse of preference fulfillment - simplified here)
        # As a battery engineer, the goal is likely efficiency/grid support over strict comfort, 
        # but we use a baseline cost reflecting required self-consumption.
        # High baseline demand suggests a baseline load must be met regardless of price/carbon.
        comfort_cost = baseline 

        # Total Composite Cost (Weighted sum)
        # We prioritize minimizing the combined environmental and economic penalty relative to the mandatory baseline energy.
        total_cost = 0.5 * price_cost + 0.3 * carbon_cost + 0.2 * comfort_cost
        
        # Constraint satisfaction bounds
        min_sessions = np.array(list(self.global_params['slot_min_sessions'].values()))
        max_sessions = np.array(list(self.global_params['slot_max_sessions'].values()))
        
        return total_cost, min_sessions, max_sessions, baseline

    def calculate_coordination_signal(self, day_key, usage):
        """
        Calculates a coordination factor based on neighbor observed behavior and global context.
        Coordination aims to smooth the load profile (minimizing peaks/valleys) relative 
        to the observed collective behavior, especially where neighbors show strong preferences.
        """
        day_data = self.scenario_data['days'][day_key]
        
        neighbor_influence = np.zeros(4)
        
        # 1. Neighbor Preference Alignment (Beta weighted)
        for nid, info in self.neighbor_info.items():
            # Assuming neighbors are trying to adhere to their preferred slots (high usage there)
            # And assuming we should counterbalance extremely high usage in slots where neighbors are concentrated,
            # or align if the coordination goal is shared (e.g., low carbon).
            
            # For simplicity in a collective scenario, we aim to smooth load peaks where neighbors are known to load up.
            # Neighbor 2 favors slots 1, 2. Neighbor 3 favors slots 1, 3. Slot 1 is highly favored.
            
            if nid == 2:
                # N2 focuses on slots 1, 2 (mid-evening)
                neighbor_influence[1] += self.beta * 0.3
                neighbor_influence[2] += self.beta * 0.3
            elif nid == 3:
                # N3 focuses on slots 1, 3 (mid-evening and late)
                neighbor_influence[1] += self.beta * 0.3
                neighbor_influence[3] += self.beta * 0.3

        # 2. Transformer/Capacity Constraint Awareness (Using global capacity context)
        # Capacity is 6.8 MW. Baseline load is around 21.9 MW total (5.2+5.0+4.9+6.5).
        # The agent capacity is not explicitly given, but we infer local constraints from spatial carbon spikes.
        # Since location 1 (Agent 1) is associated with Day 2/Day 6 warnings about balancing transformer temps,
        # we should try to avoid simultaneous peaks with neighbors if possible.
        
        # Check spatial carbon for Location 1 (Agent 1) to see expected high stress areas for this location
        spatial_carbon_str = day_data.get('Spatial carbon', {}).get(str(self.location_id))
        if spatial_carbon_str:
            spatial_carbon = np.array([float(x) for x in spatial_carbon_str.split('; ')])
            # High spatial carbon suggests high local load stress, penalize usage there unless mandated by other factors.
            stress_penalty = (spatial_carbon / np.max(spatial_carbon)) * 0.2 * self.beta 
            neighbor_influence += stress_penalty

        # Coordination signal: A higher value means 'good' usage or 'required' usage based on external factors.
        # Since we are minimizing cost, we want coordination to *reduce* the effective cost in desirable slots.
        # Coordination is an *incentive* to use the slot, so we subtract it from the cost.
        coordination_incentive = neighbor_influence
        
        return coordination_incentive

    def determine_usage(self):
        """Computes the 7-day usage matrix based on optimized utility."""
        self.load_scenario()
        
        all_days_usage = []
        
        # Pre-calculate base demand for comfort penalty
        base_demand = self.base_demand
        
        # Neighbor preferred slots (for reference during optimization)
        n2_pref = self.neighbor_info[2]['preferred_slots']
        n3_pref = self.neighbor_info[3]['preferred_slots']
        
        for day_index, day_key in enumerate(self.day_keys):
            
            total_cost, min_s, max_s, baseline = self.calculate_daily_costs(day_key)
            coordination_incentive = self.calculate_coordination_signal(day_key, None)
            
            # Effective Cost = Total Cost - Coordination Incentive
            effective_cost = total_cost - coordination_incentive
            
            # --- Optimization Heuristic: Greedy approach based on lowest effective cost ---
            
            # 1. Initialize usage to minimum required sessions (to satisfy implicit constraints)
            usage = np.copy(min_s).astype(float)
            
            # 2. Determine remaining capacity for non-mandatory usage
            remaining_capacity = max_s - min_s
            
            # 3. Calculate the "value" of adding one unit of usage in each slot
            # Value is the negative cost (i.e., how much we save/gain by using it)
            slot_value = -effective_cost
            
            # 4. Iterate slots based on cost (or value) ranking
            # We want to fill slots that have the lowest effective cost first, up to max_s.
            
            # Determine slots to fill above the minimum requirement
            slots_to_fill = sorted(range(4), key=lambda i: slot_value[i], reverse=True)
            
            # Calculate the total required energy above baseline (heuristic for total energy required)
            # Since the output must be [0, 1], we treat the base demand as a minimum *fraction* of the slot capacity (1.0).
            # We need to decide how much *more* to add above the minimum mandated sessions.
            
            # We aim for a total usage that reflects the baseline demand relative to capacity, 
            # but the output scale is normalized [0, 1]. We use the baseline load to guide proportion.
            
            # Calculate ideal normalized load relative to the largest baseline value observed across the day
            max_baseline_day = np.max(baseline)
            normalized_baseline = baseline / max_baseline_day
            
            # Target usage combines minimum required sessions with a factor of normalized baseline load, 
            # capped by max_sessions.
            
            target_usage = (min_s * 0.5 + normalized_baseline * 0.5) # Mix minimum mandate with normalized baseline importance
            
            # Refine target using effective cost gradient:
            final_usage = np.copy(target_usage)
            
            # Enforce bounds:
            final_usage = np.clip(final_usage, min_s, max_s)
            
            # 5. Final fine-tuning based on relative cost structure (if the target is too conservative or aggressive)
            # If the cost is extremely low in a slot, try to increase usage towards max_s, provided the incentive is strong.
            for i in range(4):
                if effective_cost[i] < np.min(effective_cost) * 0.8: # Very cheap slot
                    final_usage[i] = np.clip(final_usage[i] * 1.1, final_usage[i], max_s[i])
                
                # Ensure we meet minimum if clipping reduced it (shouldn't happen if target_usage uses min_s as base)
                final_usage[i] = np.clip(final_usage[i], min_s[i], max_s[i])

            
            # --- Incorporate Comfort Penalty Heuristic (Battery Engineer Profile) ---
            # As an engineer, maybe we sacrifice comfort slightly (higher usage) if the grid is stressed (high spatial carbon on *other* feeders, or low overall cost).
            # Since we are Location 1, and Day 2/6 mention balancing *our* transformer temps, we must be cautious here.
            
            # If the overall cost is low, we should try to maximize output (utility/backfeed potential).
            # Since we don't know the absolute capacity/PV, we use the normalized target as the main driver, constrained by cost.
            
            # The final output should be smooth and respect cost/coordination.
            
            all_days_usage.append(final_usage.tolist())

        # 6. Write output
        output_data = {f"Day {i+1}": all_days_usage[i] for i in range(7)}
        
        with open('global_policy_output.json', 'w') as f:
            json.dump(output_data, f, indent=4)
            
        return all_days_usage

# Execution Block
if __name__ == "__main__":
    policy = DistributedEnergyPolicy()
    policy.determine_usage()