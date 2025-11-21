import json
import numpy as np
from typing import List, Dict, Any

class Agent3Policy:
    def __init__(self, scenario_data: Dict[str, Any]):
        self.scenario = scenario_data
        self.num_slots = 4
        self.num_days = 7
        self.location_id = self.scenario['persona_attributes']['location']
        self.base_demand = np.array(self.scenario['persona_attributes']['base_demand'])

        # Extract constraints from the global structure derived from the prompt header
        self.slot_min_sessions = np.array(self.scenario['slot_min_sessions'])
        self.slot_max_sessions = np.array(self.scenario['slot_max_sessions'])

    def _get_day_specific_data(self, day_index: int) -> Dict[str, np.ndarray]:
        day_key = f"Day {day_index + 1}"
        day_data = self.scenario['days'][day_key]

        price = np.array(day_data['Tariff'])
        carbon = np.array(day_data['Carbon'])
        baseline = np.array(day_data['Baseline load'])
        
        spatial_carbon_map = day_data.get('Spatial carbon', {})
        
        spatial_carbon_str = None
        if str(self.location_id) in spatial_carbon_map:
             spatial_carbon_str = spatial_carbon_map[str(self.location_id)]
        
        spatial_carbon = carbon # Default fallback
        if spatial_carbon_str:
            try:
                spatial_carbon = np.array([float(c.strip()) for c in spatial_carbon_str.split(', ')])
            except ValueError:
                spatial_carbon = carbon

        return {
            "price": price,
            "carbon": carbon,
            "baseline": baseline,
            "spatial_carbon": spatial_carbon
        }

    def evaluate_slot(self, day_index: int, slot_index: int) -> float:
        """Calculates a composite cost/penalty score (lower is better)."""
        data = self._get_day_specific_data(day_index)
        
        price_score = data['price'][slot_index]
        carbon_score = data['carbon'][slot_index] 
        spatial_score = data['spatial_carbon'][slot_index]
        
        # Persona: Nurse (Location 3), Base demand profile: [0.60, 0.80, 0.90, 0.70]
        
        # Weighting based on persona needs: Prioritize slots with high base demand when costs are similar.
        persona_priority_factor = 1.0
        if slot_index == 2: # Slot 21-22 (Base 0.90) -> Highest priority, lowest cost factor
            persona_priority_factor = 0.5 
        elif slot_index == 1: # Slot 20-21 (Base 0.80)
            persona_priority_factor = 0.7
        elif slot_index == 3: # Slot 22-23 (Base 0.70)
            persona_priority_factor = 1.0
        elif slot_index == 0: # Slot 19-20 (Base 0.60) -> Lowest priority
            persona_priority_factor = 1.5 
            
        # Normalization constants (based on general scale)
        norm_price = price_score / 0.35 
        norm_carbon = carbon_score / 700.0 
        norm_spatial = spatial_score / 650.0 
        
        # Score calculation: Heavily weight price and carbon, slightly weight spatial for local awareness
        composite_score = (1.5 * norm_price) + (1.0 * norm_carbon) + (0.5 * norm_spatial)
        
        # Apply persona factor: Lower factor means higher overall charging priority
        final_score = composite_score / persona_priority_factor
        
        return final_score

    def calculate_daily_usage(self, day_index: int) -> np.ndarray:
        
        # 1. Calculate ranking scores
        scores_for_ranking = np.array([self.evaluate_slot(day_index, s) for s in range(self.num_slots)])
        
        # 2. Transform scores into utilization factors (0 to 1), where low score -> high factor
        max_score = np.max(scores_for_ranking)
        min_score = np.min(scores_for_ranking)
        
        if max_score > min_score:
            # Utilization is inversely proportional to the normalized score rank
            utilization_factors = 1.0 - (scores_for_ranking - min_score) / (max_score - min_score)
        else:
            utilization_factors = np.full(self.num_slots, 0.5)
            
        # 3. Combine utilization factor with persona base demand profile
        weighted_usage = utilization_factors * self.base_demand 
        
        # 4. Normalize to a total usage sum (we normalize such that the distribution matches base demand weighted by cost avoidance)
        total_weighted = np.sum(weighted_usage)
        
        # We target the shape defined by weighted_usage, capped at 1.0 per slot.
        final_usage = np.clip(weighted_usage, 0.0, 1.0)
        
        # Re-normalize the shape if the resulting vector sum is very small, ensuring we don't output all zeros
        if np.sum(final_usage) < 0.01:
             final_usage = self.base_demand / np.sum(self.base_demand) * 0.5 # Default to moderate distribution if costs are too high everywhere

        # 5. Apply session constraints as floors (Crucial for meeting minimum requirements)
        S_min = self.slot_min_sessions 
        S_max = self.slot_max_sessions 
        
        for s in range(self.num_slots):
            if S_min[s] > 0 and S_max[s] > 0:
                # Enforce minimum share based on session ratio, relative to the overall demand scale (0.5 proxy)
                min_share_required = (S_min[s] / S_max[s]) * self.base_demand[s] * 1.2 
                final_usage[s] = max(final_usage[s], min_share_required)
            
            # Apply explicit day constraints if known (Day 6, Slot 2 rationed)
            day_key = f"Day {day_index + 1}"
            if day_key == "Day 6" and s == 2:
                final_usage[s] = min(final_usage[s], 0.4) # Explicitly cap Slot 2 on Day 6

            final_usage[s] = min(final_usage[s], 1.0)

        # Final small adjustment to ensure total shape is sensible, favoring the cost structure over rigid sum=1.0
        # If the usage is heavily skewed, we might need to redistribute a small amount, but clipping/flooring is prioritized.
        if np.sum(final_usage) > 4.0: # Should never happen due to min(1.0)
             final_usage = np.clip(final_usage, 0.0, 1.0)
        
        return np.round(final_usage, 3)


    def generate_policy(self) -> List[List[float]]:
        policy_output = []
        for day_index in range(self.num_days):
            daily_usage = self.calculate_daily_usage(day_index)
            policy_output.append(daily_usage.tolist())
        return policy_output

def main():
    try:
        with open("scenario.json", 'r') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        return

    agent = Agent3Policy(scenario_data)
    policy = agent.generate_policy()

    with open("local_policy_output.json", 'w') as f:
        json.dump(policy, f, indent=4)

if __name__ == "__main__":
    main()