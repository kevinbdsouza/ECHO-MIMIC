import json

class PolicyGenerator:
    def __init__(self):
        # Scenario context derived from the prompt for Agent 2 (Feeder Analyst, Location 2)
        self.context = {
            "slots": 4,
            "capacity": 6.8,
            "persona": "Position 2 feeder analyst prioritising transformer headroom",
            "location": 2,
            "base_demand": [0.70, 1.00, 0.80, 0.50],
            "forecast_tariff": [0.23, 0.24, 0.27, 0.30],
            "forecast_carbon": [700, 480, 500, 750],
            "forecast_baseline": [5.2, 5.0, 4.9, 6.5],
            "slot_min_sessions": [1, 1, 1, 1],
            "slot_max_sessions": [2, 2, 1, 2],
            "days": {
                "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5]},
                "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6]},
                "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4]},
                "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7]},
                "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6]},
                "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5]},
                "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3]},
            }
        }
        # Location specific spatial carbon data for Agent 2 (Location 2)
        self.spatial_carbon_data = {
            "Day 1": [550, 340, 520, 600],
            "Day 2": [540, 500, 320, 610],
            "Day 3": [320, 510, 540, 600],
            "Day 4": [550, 330, 520, 580],
            "Day 5": [560, 500, 320, 590],
            "Day 6": [320, 510, 560, 620],
            "Day 7": [540, 330, 520, 600],
        }
        self.day_names = list(self.context["days"].keys())
        
        # Pre-calculate max observed values for normalization safety
        self.MAX_TARIFF = 0.32
        self.MAX_CARBON = 750
        
        # Pre-calculate max spatial carbon across all days for each slot to normalize stress impact
        self.MAX_SPATIAL_PER_SLOT = [
            max(self.spatial_carbon_data[d][s] for d in self.day_names) for s in range(4)
        ]


    def calculate_usage(self):
        all_day_usages = []
        base_demand = self.context["base_demand"]
        
        for day_idx, day_name in enumerate(self.day_names):
            day_data = self.context["days"][day_name]
            local_spatial_carbon = self.spatial_carbon_data[day_name]
            
            tariffs = day_data["Tariff"]
            carbon = day_data["Carbon"]
            
            day_usage = [0.0] * 4
            
            # Primary Goal: Maximize headroom -> Minimize spatial stress / baseline load impact
            # Secondary Goal: Fulfill base demand (Slot 1 > Slot 2 > Slot 0 > Slot 3)
            
            # Score calculation: Lower score is better (less undesirable)
            stress_scores = []
            for s in range(4):
                
                # 1. Spatial Stress (Primary driver for Feeder Analyst)
                norm_sc = local_spatial_carbon[s] / self.MAX_SPATIAL_PER_SLOT[s]
                
                # 2. Cost/Carbon (Secondary driver)
                norm_tariff = tariffs[s] / self.MAX_TARIFF
                norm_carbon = carbon[s] / self.MAX_CARBON
                
                # Composite Score (Minimize stress, prefer cheap/green)
                score = (0.6 * norm_sc) + (0.3 * norm_tariff) + (0.1 * norm_carbon)
                stress_scores.append(score)
            
            # Transform score into a priority metric (Higher is better for charging)
            # Normalize scores relative to the average score for this day
            avg_score = sum(stress_scores) / 4.0
            priority_metric = [max(0.0, 1.0 - (score / avg_score)) for score in stress_scores]
            
            # Adjust usage based on priority metric, while anchoring around base demand
            temp_usage = [0.0] * 4
            for s in range(4):
                
                # Anchor usage to base demand, modulated by the priority metric
                # Priority metric close to 1.0 means we heavily favor charging here relative to baseline expectation
                # Priority metric close to 0.0 means we suppress charging here relative to baseline expectation
                
                # Weighting: 70% by base need, 30% by observed priority
                
                # If priority is high (score is low), boost usage above base demand (up to 1.0)
                if priority_metric[s] > 1.1: # Significantly better than average
                    temp_usage[s] = min(1.0, base_demand[s] * 1.2)
                # If priority is low (score is high), suppress usage relative to base demand
                elif priority_metric[s] < 0.9:
                    temp_usage[s] = max(0.0, base_demand[s] * 0.8)
                else:
                    # Moderate priority, stick close to base demand
                    temp_usage[s] = base_demand[s]
            
            # --- Apply Explicit Day Constraints ---
            
            # Day 6: Slot 2 rationed (Maintenance advisory)
            if day_name == "Day 6":
                temp_usage[2] = min(temp_usage[2], 0.15) # Severe reduction from base 0.8
            
            # Day 4: Staggering (Avoid high usage in Slot 0 due to neighbor congestion, Slot 3 is inherently low)
            if day_name == "Day 4":
                temp_usage[0] = min(temp_usage[0], 0.6)
                
            # Day 5: High carbon spike in Slot 3 (post 22h)
            if day_name == "Day 5":
                temp_usage[3] = min(temp_usage[3], 0.3)

            # Final clamp to [0, 1]
            day_usage = [max(0.0, min(1.0, u)) for u in temp_usage]
            
            # Enforce minimum session compliance (if slot_min_sessions=1 implies usage > 0.1)
            min_session_floor = 0.1
            for s in range(4):
                 if self.context["slot_min_sessions"][s] >= 1:
                     day_usage[s] = max(day_usage[s], min_session_floor)

            all_day_usages.append(day_usage)

        return all_day_usages

    def generate_policy(self):
        usage_plan = self.calculate_usage()
        
        # Save to local_policy_output.json
        with open("local_policy_output.json", "w") as f:
            json.dump(usage_plan, f, indent=4)


if __name__ == "__main__":
    generator = PolicyGenerator()
    generator.generate_policy()