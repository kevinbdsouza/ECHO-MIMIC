import json
import numpy as np
from typing import List, Dict, Any

class AgentPolicy:
    """
    Agent 1: Battery engineer balancing budget and solar backfeed.
    Focuses on low cost and low carbon, prioritizing times when their local solar 
    contribution (implied by low local carbon/price) is most effective, while 
    meeting base demand. Location 1.
    """
    def __init__(self, persona_data: Dict[str, Any]):
        self.location = persona_data['location']
        self.base_demand = np.array(persona_data['base_demand'])
        self.slot_mins = np.array([1, 1, 1, 1])
        self.slot_maxs = np.array([2, 2, 1, 2])
        
        # Initial cost/carbon weights (can be refined based on goals)
        # Lower tariff and lower carbon are highly preferred.
        self.COST_WEIGHT = 0.5
        self.CARBON_WEIGHT = 0.5
        
        # Day 6 constraint: slot 2 is rationed due to maintenance advisory.
        self.day6_ration_slot2 = 0.7 # Reduce usage slightly in slot 2 on Day 6

    def calculate_day_score(self, day_data: Dict[str, Any]) -> np.ndarray:
        tariffs = np.array(day_data['Tariff'])
        carbons = np.array(day_data['Carbon'])
        
        # Spatial carbon for location 1 is the first element in the spatial string
        spatial_carbon_str = day_data['Spatial carbon'][self.location - 1]
        spatial_carbons = np.array([int(s) for s in spatial_carbon_str.split(', ')])

        # Combine price, carbon, and spatial carbon into a composite cost metric
        # Lower is better.
        # Weighting spatial carbon heavily as the 'solar backfeed' concern often relates to local grid state.
        composite_cost = (
            self.COST_WEIGHT * tariffs + 
            self.CARBON_WEIGHT * carbons + 
            (1 - self.CARBON_WEIGHT) * spatial_carbons
        )
        
        # Normalize cost inversely to create a preference score (higher is better)
        # Use a simple inverse transformation relative to max possible cost to keep scores positive and meaningful.
        MAX_POSSIBLE_COST = 1000 # Rough upper bound estimate for the scaled metrics
        preference_score = 1.0 - (composite_cost / np.max(composite_cost))

        # Apply base demand scaling (implicitly, we want to meet this demand, so slots where demand is high are naturally favored if costs are okay)
        # Since base_demand is a requirement, we will use the preference score directly to allocate the *extra* above base, 
        # but for simplicity here, we'll use the score to determine the relative usage across slots.
        
        return preference_score

    def generate_usage_plan(self, scenario_data: Dict[str, Any]) -> List[List[float]]:
        
        policy_output = []
        day_keys = [k for k in scenario_data.keys() if k.startswith("Day")]
        
        # The fixed 4 slots defined at the start
        num_slots = 4
        
        for day_index, day_name in enumerate(day_keys):
            day_data = scenario_data[day_name]
            
            # 1. Calculate inherent preferences based on external factors
            preferences = self.calculate_day_score(day_data)
            
            # 2. Determine target total energy needed relative to capacity constraint
            # We must cover the base demand, assuming capacity dictates the maximum total energy.
            
            # Total required energy (normalized by an assumed 1 unit energy per 1 unit usage fraction)
            # We must always meet at least the normalized base demand.
            total_base_demand = np.sum(self.base_demand)
            
            # Determine the total fraction of energy to use based on fairness/total availability.
            # Since no explicit total demand is given other than base_demand, we aim to meet the base demand first,
            # and then scale usage based on preferences, capped by slot limits [0, 1].
            
            # Normalized usage based purely on preference, respecting [0, 1] bounds initially
            # Add a small base value to ensure minimum usage if preferences are near zero
            base_usage = 0.2 
            normalized_usage = np.clip(preferences + base_usage, 0.0, 1.0)
            
            # Apply Day 6 specific constraint
            if day_index + 1 == 6:
                normalized_usage[2] *= self.day6_ration_slot2


            # 3. Scale usage to meet the base demand requirement (must be at least base_demand scaled to [0, 1] usage)
            # Since the prompt requires output in [0, 1] and provides base_demand, we must ensure usage >= base_demand * some_scaling_factor.
            # If we assume the base_demand vector provided (e.g., 1.2, 0.7, 0.8, 0.6) represents the required *fraction* of capacity
            # needed in that slot to satisfy the agent's minimal needs (and these values are often > 1.0 in the prompt, which is confusing 
            # as usage must be <= 1.0), we must interpret:
            # Requirement: usage[i] >= min(1.0, base_demand[i] * K) where K scales the base demand to the actual usage scale [0, 1].
            
            # Interpretation based on constraints: Usage must be in [0, 1]. Base demand is what MUST be met. 
            # If base_demand > 1.0, we must clip it to 1.0, effectively meaning 100% usage in that slot is mandatory.
            
            # Let's assume base_demand implies the minimum required usage fraction, capped at 1.0.
            min_required_usage = np.clip(self.base_demand, 0.0, 1.0)
            
            # The final usage is the maximum of the preferred usage and the minimum required usage.
            final_usage = np.maximum(normalized_usage, min_required_usage)
            
            # 4. Apply slot session limits (These limits [min_sessions, max_sessions] are usually for the *number* of sessions, 
            # not the usage fraction [0, 1]. Since the output must be usage fraction, we treat them as soft constraints on fraction, 
            # or assume the usage fraction maps linearly to session count if the charging power is constant across sessions.)
            
            # Since we have no mapping from fraction [0,1] to session count, we primarily use slot_maxs as a hard cap of 1.0, 
            # and ignore slot_mins as they conflict with the explicit usage requirement [0, 1] unless base_demand handles them.
            
            # Slot max sessions suggest max usage of 1.0 is generally safe unless constrained by capacity/cost.
            # Given the explicit slot_max_sessions (e.g., slot 2 max 1 session), we interpret this as a hard cap of 1.0 usage anyway.
            
            # Ensure final usage respects the hard cap of 1.0
            final_usage = np.clip(final_usage, 0.0, 1.0)
            
            policy_output.append(final_usage.tolist())
            
        return policy_output

def main():
    # Simulate loading scenario data from the context provided in the prompt
    # In a real environment, this would load scenario.json from disk.
    scenario_context = {
        "scenario_id": "ev_peak_sharing_1",
        "slots": "0: 19-20, 1: 20-21, 2: 21-22, 3: 22-23",
        "price": "0.23, 0.24, 0.27, 0.30",
        "carbon_intensity": "700, 480, 500, 750",
        "capacity": 6.8,
        "baseline_load": "5.2, 5.0, 4.9, 6.5",
        "slot_min_sessions": "0: 1, 1: 1, 2: 1, 3: 1",
        "slot_max_sessions": "0: 2, 1: 2, 2: 1, 3: 2",
        "spatial_carbon": "1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | 3: 503, 473, 471, 577 | 4: 617, 549, 479, 363 | 5: 411, 376, 554, 623",
        "days": {
            "Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)": {
                "Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], 
                "Baseline load": [5.3, 5.0, 4.8, 6.5], 
                "Spatial carbon": ["330, 520, 560, 610", "550, 340, 520, 600", "590, 520, 340, 630", "620, 560, 500, 330", "360, 380, 560, 620"]
            },
            "Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)": {
                "Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], 
                "Baseline load": [5.1, 5.2, 4.9, 6.6], 
                "Spatial carbon": ["510, 330, 550, 600", "540, 500, 320, 610", "310, 520, 550, 630", "620, 540, 500, 340", "320, 410, 560, 640"]
            },
            "Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)": {
                "Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], 
                "Baseline load": [5.4, 5.0, 4.9, 6.4], 
                "Spatial carbon": ["540, 500, 320, 600", "320, 510, 540, 600", "560, 330, 520, 610", "620, 560, 500, 330", "330, 420, 550, 640"]
            },
            "Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)": {
                "Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], 
                "Baseline load": [5.0, 5.1, 5.0, 6.7], 
                "Spatial carbon": ["320, 520, 560, 600", "550, 330, 520, 580", "600, 540, 500, 320", "560, 500, 330, 540", "500, 340, 560, 630"]
            },
            "Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)": {
                "Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], 
                "Baseline load": [5.2, 5.3, 5.0, 6.6], 
                "Spatial carbon": ["510, 330, 560, 600", "560, 500, 320, 590", "320, 520, 540, 620", "630, 560, 510, 340", "330, 420, 560, 630"]
            },
            "Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)": {
                "Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], 
                "Baseline load": [5.5, 5.2, 4.8, 6.5], 
                "Spatial carbon": ["540, 500, 320, 610", "320, 510, 560, 620", "560, 340, 520, 610", "640, 560, 510, 330", "520, 330, 540, 600"]
            },
            "Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)": {
                "Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], 
                "Baseline load": [5.1, 4.9, 4.8, 6.3], 
                "Spatial carbon": ["330, 520, 560, 610", "540, 330, 520, 600", "580, 540, 330, 620", "630, 560, 500, 330", "520, 330, 550, 600"]
            }
        }
    }
    
    persona_data = {
        "location": 1,
        "base_demand": [1.20, 0.70, 0.80, 0.60]
    }
    
    agent = AgentPolicy(persona_data)
    policy = agent.generate_usage_plan(scenario_context['days'])
    
    # 3. Write local_policy_output.json
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(policy, f, indent=4)

if __name__ == "__main__":
    main()