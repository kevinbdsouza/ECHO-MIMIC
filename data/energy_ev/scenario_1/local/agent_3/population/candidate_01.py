import json

class AgentPolicy:
    def __init__(self):
        # Agent Persona & Objectives
        self.location = 3
        # Base demand: [19-20, 20-21, 21-22, 22-23]
        self.base_demand = [0.60, 0.80, 0.90, 0.70]
        
        # Day-specific data structure (extracted from prompt context)
        self.scenario_data = {
            "Day 1": {
                "Tariff": [0.20, 0.25, 0.29, 0.32],
                "Carbon": [490, 470, 495, 540],
                "Baseline load": [5.3, 5.0, 4.8, 6.5],
                # Spatial carbon for location 3 (index 2 in the list)
                "Spatial carbon": [590, 520, 340, 630] 
            },
            "Day 2": {
                "Tariff": [0.27, 0.22, 0.24, 0.31],
                "Carbon": [485, 460, 500, 545],
                "Baseline load": [5.1, 5.2, 4.9, 6.6],
                "Spatial carbon": [310, 520, 550, 630]
            },
            "Day 3": {
                "Tariff": [0.24, 0.21, 0.26, 0.30],
                "Carbon": [500, 455, 505, 550],
                "Baseline load": [5.4, 5.0, 4.9, 6.4],
                "Spatial carbon": [560, 330, 520, 610]
            },
            "Day 4": {
                "Tariff": [0.19, 0.24, 0.28, 0.22],
                "Carbon": [495, 470, 500, 535],
                "Baseline load": [5.0, 5.1, 5.0, 6.7],
                "Spatial carbon": [600, 540, 500, 320]
            },
            "Day 5": {
                "Tariff": [0.23, 0.20, 0.27, 0.31],
                "Carbon": [500, 450, 505, 545],
                "Baseline load": [5.2, 5.3, 5.0, 6.6],
                "Spatial carbon": [320, 520, 540, 620]
            },
            "Day 6": {
                "Tariff": [0.26, 0.22, 0.25, 0.29],
                "Carbon": [505, 460, 495, 540],
                "Baseline load": [5.5, 5.2, 4.8, 6.5],
                "Spatial carbon": [560, 340, 520, 610]
            },
            "Day 7": {
                "Tariff": [0.21, 0.23, 0.28, 0.26],
                "Carbon": [495, 460, 500, 530],
                "Baseline load": [5.1, 4.9, 4.8, 6.3],
                "Spatial carbon": [580, 540, 330, 620]
            }
        }
        
        # Slot constraints (derived from scenario header)
        self.slot_min_sessions = [1, 1, 1, 1]
        self.slot_max_sessions = [2, 2, 1, 2]
        self.capacity = 6.8 # Not directly used for individual agents unless total demand is calculated, but contextually important.

    def calculate_scores(self, day_data):
        """Calculates a composite score for minimization (lower is better)."""
        
        tariffs = day_data["Tariff"]
        carbons = day_data["Carbon"]
        # Spatial carbon for agent location 3
        spatial_carbons = day_data["Spatial carbon"] 
        
        scores = []
        for i in range(4):
            # Persona: Night-shift nurse (Location 3). 
            # Primary concern is meeting personal demand (base_demand), 
            # and avoiding local/spatial spikes which might imply network strain near the hospital/residence area.
            
            # Cost function: Weighted sum of Tariff, Carbon Intensity, and Spatial Carbon (local strain)
            # Using slightly higher weights for local spatial carbon, as nurse might be sensitive to local network issues.
            score = (
                tariffs[i] * 1.0 +  # Price sensitivity
                carbons[i] * 0.001 + # Carbon sensitivity (scaled down)
                spatial_carbons[i] * 0.002 # Local spatial sensitivity
            )
            scores.append(score)
        return scores

    def generate_usage_plan(self):
        all_days_usage = []
        
        day_keys = list(self.scenario_data.keys())
        
        for day_name in day_keys:
            day_data = self.scenario_data[day_name]
            base_demand = self.base_demand
            
            # 1. Calculate scores (lower is better)
            scores = self.calculate_scores(day_data)
            
            # 2. Determine relative charging preference based on scores and base demand
            raw_usage = [0.0] * 4
            
            # Combine base demand and inverse score preference (we want high base demand slots 
            # that also have low scores)
            
            # Normalize scores to [0, 1] where 0 is best
            min_score = min(scores)
            max_score = max(scores)
            if max_score - min_score > 1e-6:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [0.0] * 4

            # Preference = Base Demand * (1 - Normalized Score)
            # High base demand and low normalized score yields high preference.
            for i in range(4):
                preference = base_demand[i] * (1.0 - normalized_scores[i])
                raw_usage[i] = preference
            
            # 3. Apply constraints and normalize to [0, 1]
            final_usage = [0.0] * 4
            
            # Simple scaling: Ensure we meet the spirit of the base demand/preference while staying in [0, 1]
            # Since we cannot easily model total energy required, we scale the relative preferences.
            
            # We must respect min/max session constraints conceptually by ensuring usage is high 
            # where base demand is high, but capping at 1.0.
            
            # First pass: Clip to [0, 1] and respect min usage if the preference is high enough
            
            # A simple strategy for fixed relative charging: Scale the raw preference vector 
            # so that the highest preferred slot hits 1.0, but ensure we place *something* 
            # if the base demand requires it, respecting the relative costs.
            
            max_raw = max(raw_usage)
            if max_raw > 1e-6:
                scaled_usage = [u / max_raw for u in raw_usage]
            else:
                scaled_usage = [0.0] * 4

            # Second pass: Re-incorporate base demand expectations and constraints loosely.
            # Since we have no actual KWh target, we assume base_demand dictates the *relative* shape,
            # and the optimization dictates the *timing*. We will prioritize the shape dictated by cost, 
            # scaled to fit [0, 1], ensuring we cover the slots where base demand is high.
            
            
            # Strategy Refined: Use the inverse score as the primary driver for energy distribution, 
            # but ensure slots with high base demand are prioritized slightly higher if costs are close.
            
            # Use inverse normalized score (lower score = higher inverse_score [1..0])
            inverse_normalized_scores = [1.0 - n for n in normalized_scores]
            
            # Combined weighted factor: Favor slots that are cheap AND have high base demand.
            weighted_factor = [
                inverse_normalized_scores[i] * base_demand[i] 
                for i in range(4)
            ]
            
            max_factor = max(weighted_factor)
            
            if max_factor > 1e-6:
                usage = [w / max_factor for w in weighted_factor]
            else:
                usage = [0.0] * 4
            
            # Final clamp and ensuring minimum feasibility (even if abstractly imposed by base demand structure)
            for i in range(4):
                # Clamp to [0, 1]
                final_usage[i] = max(0.0, min(1.0, usage[i]))
                
                # Apply Day 6 specific constraint (Slot 2 rationed). 
                # This is an external constraint not captured by our costs, must be hardcoded if known.
                if day_name == "Day 6" and i == 2:
                     # Assuming rationing means reducing usage significantly below preference/demand
                    final_usage[i] *= 0.4 

            # Ensure usage is sensible given the implicit minimum session requirement (we cannot verify
            # if the resulting usage corresponds to 1 session, so we ensure it's not near zero if demand is high)
            for i in range(4):
                 if self.base_demand[i] > 0.7 and final_usage[i] < 0.1:
                     final_usage[i] = min(1.0, self.base_demand[i] * 0.2) # Small boost if highly demanded slot is ignored

            all_days_usage.append(final_usage)
            
        return all_days_usage

    def save_policy(self, usage_data):
        output_filename = "local_policy_output.json"
        with open(output_filename, 'w') as f:
            json.dump(usage_data, f, indent=4)
        # print(f"Policy saved to {output_filename}")

if __name__ == "__main__":
    agent = AgentPolicy()
    policy_output = agent.generate_usage_plan()
    agent.save_policy(policy_output)