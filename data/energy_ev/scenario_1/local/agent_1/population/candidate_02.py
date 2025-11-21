import json

class Policy:
    def __init__(self, scenario_data, profile):
        self.scenario = scenario_data
        self.profile = profile
        self.num_slots = 4
        self.num_days = 7
        self.capacity = self.scenario['capacity']
        self.base_demand = self.profile['base_demand']
        self.total_base_energy = sum(self.base_demand)
        self.location_id = self.profile['location']

    def calculate_usage(self):
        # Persona: Battery Engineer (Location 1) balancing budget and solar backfeed.
        # Strategy: Prioritize low cost/carbon (especially low spatial carbon for backfeed alignment),
        # ensure base demand is covered, and apply known day-specific constraints.
        
        policy_output = []
        days_data = self.scenario['days']
        day_keys = list(days_data.keys())
        
        for day_index in range(self.num_days):
            day_key = day_keys[day_index]
            day_info = days_data[day_key]
            
            day_tariffs = day_info['Tariff']
            day_carbons = day_info['Carbon']
            
            # Extract spatial carbon for location 1 (self)
            # Spatial carbon data is structured as "1: C0, C1, C2, C3; 2: ...; 5: ..."
            # We must parse the correct location string based on the raw input format structure used in the prompt for day data.
            spatial_carbon_str = day_info['Spatial carbon']
            
            # Find the string segment corresponding to location 1
            loc_carbon_segment = None
            for segment in spatial_carbon_str.split('; '):
                if segment.startswith(f"{self.location_id}:"):
                    loc_carbon_segment = segment.split(':')[1].strip()
                    break
            
            if not loc_carbon_segment:
                 # Fallback: Use average/default if parsing fails due to unexpected format shift
                 day_spatial_carbons = [500] * self.num_slots
            else:
                 day_spatial_carbons = [int(c) for c in loc_carbon_segment.split(', ')]


            scores = []
            for slot in range(self.num_slots):
                price = day_tariffs[slot]
                carbon = day_carbons[slot]
                spatial_carbon = day_spatial_carbons[slot]
                
                # Score: Lower is better. Weighting: Price (1.0), Carbon (0.002), Spatial Carbon (0.001)
                score = (1.0 * price) + (0.002 * carbon) + (0.001 * spatial_carbon)
                scores.append(score)

            # 1. Determine desirability based on inverse cost
            min_score = min(scores)
            max_score = max(scores)
            
            # Map score to desirability factor (higher is better)
            if max_score - min_score < 1e-6:
                desirability = [1.0] * self.num_slots
            else:
                # Inverse mapping, offset to ensure positive desirability
                desirability = [(max_score - s) + (max_score - min_score) * 0.1 for s in scores]
            
            total_desirability = sum(desirability)
            normalized_allocation = [d / total_desirability for d in desirability]
            
            slot_usage = []
            for slot in range(self.num_slots):
                preference_level = normalized_allocation[slot]
                base_ratio = self.base_demand[slot] / self.total_base_energy
                
                # Blend: 75% preference driven by cost/carbon, 25% anchored by relative base demand requirement
                intermediate_usage = 0.75 * preference_level + 0.25 * base_ratio
                
                final_usage = max(0.0, min(1.0, intermediate_usage))
                
                # Apply Day 6 specific constraint (Slot 2 rationing)
                if day_key == 'Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)' and slot == 2:
                     final_usage = min(final_usage, 0.8) 
                     
                slot_usage.append(final_usage)
            
            policy_output.append(slot_usage)

        return policy_output

    def generate_policy(self):
        usage_vectors = self.calculate_usage()
        return usage_vectors

def main():
    # Load scenario data structure directly from prompt context for guaranteed execution
    scenario_data = {
        "scenario_id": "ev_peak_sharing_1",
        "slots": {0: '19-20', 1: '20-21', 2: '21-22', 3: '22-23'},
        "price": [0.23, 0.24, 0.27, 0.30],
        "carbon_intensity": [700, 480, 500, 750],
        "capacity": 6.8,
        "baseline_load": [5.2, 5.0, 4.9, 6.5],
        "slot_min_sessions": {0: 1, 1: 1, 2: 1, 3: 1},
        "slot_max_sessions": {0: 2, 1: 2, 2: 1, 3: 2},
        "spatial_carbon": "1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | 3: 503, 473, 471, 577 | 4: 617, 549, 479, 363 | 5: 411, 376, 554, 623",
        "days": {
            'Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)': {'Tariff': [0.20, 0.25, 0.29, 0.32], 'Carbon': [490, 470, 495, 540], 'Baseline load': [5.3, 5.0, 4.8, 6.5], 'Spatial carbon': '1: 330, 520, 560, 610; 2: 550, 340, 520, 600; 3: 590, 520, 340, 630; 4: 620, 560, 500, 330; 5: 360, 380, 560, 620'},
            'Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)': {'Tariff': [0.27, 0.22, 0.24, 0.31], 'Carbon': [485, 460, 500, 545], 'Baseline load': [5.1, 5.2, 4.9, 6.6], 'Spatial carbon': '1: 510, 330, 550, 600; 2: 540, 500, 320, 610; 3: 310, 520, 550, 630; 4: 620, 540, 500, 340; 5: 320, 410, 560, 640'},
            'Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)': {'Tariff': [0.24, 0.21, 0.26, 0.30], 'Carbon': [500, 455, 505, 550], 'Baseline load': [5.4, 5.0, 4.9, 6.4], 'Spatial carbon': '1: 540, 500, 320, 600; 2: 320, 510, 540, 600; 3: 560, 330, 520, 610; 4: 620, 560, 500, 330; 5: 330, 420, 550, 640'},
            'Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)': {'Tariff': [0.19, 0.24, 0.28, 0.22], 'Carbon': [495, 470, 500, 535], 'Baseline load': [5.0, 5.1, 5.0, 6.7], 'Spatial carbon': '1: 320, 520, 560, 600; 2: 550, 330, 520, 580; 3: 600, 540, 500, 320; 4: 560, 500, 330, 540; 5: 500, 340, 560, 630'},
            'Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)': {'Tariff': [0.23, 0.20, 0.27, 0.31], 'Carbon': [500, 450, 505, 545], 'Baseline load': [5.2, 5.3, 5.0, 6.6], 'Spatial carbon': '1: 510, 330, 560, 600; 2: 560, 500, 320, 590; 3: 320, 520, 540, 620; 4: 630, 560, 510, 340; 5: 330, 420, 560, 630'},
            'Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)': {'Tariff': [0.26, 0.22, 0.25, 0.29], 'Carbon': [505, 460, 495, 540], 'Baseline load': [5.5, 5.2, 4.8, 6.5], 'Spatial carbon': '1: 540, 500, 320, 610; 2: 320, 510, 560, 620; 3: 560, 340, 520, 610; 4: 640, 560, 510, 330; 5: 520, 330, 540, 600'},
            'Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)': {'Tariff': [0.21, 0.23, 0.28, 0.26], 'Carbon': [495, 460, 500, 530], 'Baseline load': [5.1, 4.9, 4.8, 6.3], 'Spatial carbon': '1: 330, 520, 560, 610; 2: 540, 330, 520, 600; 3: 580, 540, 330, 620; 4: 630, 560, 500, 330; 5: 520, 330, 550, 600'}
        }
    }

    persona_data = {
        "persona": "Position 1 battery engineer balancing budget and solar backfeed",
        "location": 1,
        "base_demand": [1.20, 0.70, 0.80, 0.60]
    }

    policy_generator = Policy(scenario_data, persona_data)
    policy = policy_generator.generate_policy()

    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(policy, f, indent=4)

if __name__ == "__main__":
    main()