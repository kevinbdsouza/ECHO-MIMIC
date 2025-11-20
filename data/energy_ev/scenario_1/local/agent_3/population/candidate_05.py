import json

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbors, forecast):
        self.persona = persona
        self.location = location
        self.base_demand = base_demand
        self.neighbors = neighbors
        self.forecast = forecast
        self.num_slots = len(forecast['price'])
        self.num_days = len(forecast['days'])
        self.slot_indices = list(range(self.num_slots))

    def determine_slot_for_day(self, day_name, day_data, spatial_carbon_data):
        """
        Chooses the best slot based on the imitation objective (lowest combined cost: Price + light SC weight).
        Agent 3 (Nurse) likely needs charging late (slots 1, 2 based on base_demand), but in imitation, 
        pure cost minimization based on public data (Price + location SC) dominates.
        """
        
        def calculate_cost_imitation(day_data, spatial_carbon, slot_index):
            price = day_data['Tariff'][slot_index]
            # Weight spatial carbon lightly (0.001) to influence choice slightly towards low local carbon/cost
            sc = spatial_carbon[slot_index]
            return price + 0.001 * sc

        costs = []
        for i in self.slot_indices:
            cost = calculate_cost_imitation(day_data, spatial_carbon_data, i)
            
            # Explicitly penalize rationed slot on Day 6 as an observed constraint
            if day_name == 'Day 6' and i == 2:
                cost *= 100.0 

            costs.append((cost, i))
        
        costs.sort()
        best_slot_index = costs[0][1]
        
        return best_slot_index

    def generate_plan(self):
        plan = []
        day_names = list(self.forecast['days'].keys())
        
        for i in range(self.num_days):
            day_name = day_names[i]
            day_data = self.forecast['days'][day_name]
            
            # Extract spatial carbon for Agent Location 3
            spatial_carbon_data = []
            try:
                location_key = str(self.location)
                # Robust parsing of the string format "N: c0, c1, c2, c3; ..."
                sc_str_parts = day_data['Spatial carbon'].split(f'{location_key}: ')
                if len(sc_str_parts) > 1:
                    sc_str = sc_str_parts[1].split(';')[0]
                    spatial_carbon_data = [int(x) for x in sc_str.split(', ')]
            except Exception:
                # Fallback to forecast default structure if daily parsing fails
                spatial_carbon_data = self.forecast['spatial_carbon'][self.location - 1][1:]

            best_slot_index = self.determine_slot_for_day(day_name, day_data, spatial_carbon_data)
            plan.append(best_slot_index)

        return plan

def main():
    # 1. Load scenario context (Simulated load from prompt, as required by constraints)
    SCENARIO_CONTEXT = {
        'slots': {0: '19-20', 1: '20-21', 2: '21-22', 3: '22-23'},
        'price': [0.23, 0.24, 0.27, 0.30],
        'carbon_intensity': [700, 480, 500, 750],
        'capacity': 6.8,
        'baseline_load': [5.2, 5.0, 4.9, 6.5],
        'slot_min_sessions': [1, 1, 1, 1],
        'slot_max_sessions': [2, 2, 1, 2],
        'spatial_carbon': [
            [1, 440, 460, 490, 604],
            [2, 483, 431, 471, 600],
            [3, 503, 473, 471, 577],
            [4, 617, 549, 479, 363],
            [5, 411, 376, 554, 623]
        ],
        'days': {
            "Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)": {
                "Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540],
                "Baseline load": [5.3, 5.0, 4.8, 6.5],
                "Spatial carbon": '1: 330, 520, 560, 610; 2: 550, 340, 520, 600; 3: 590, 520, 340, 630; 4: 620, 560, 500, 330; 5: 360, 380, 560, 620'
            },
            "Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)": {
                "Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545],
                "Baseline load": [5.1, 5.2, 4.9, 6.6],
                "Spatial carbon": '1: 510, 330, 550, 600; 2: 540, 500, 320, 610; 3: 310, 520, 550, 630; 4: 620, 540, 500, 340; 5: 320, 410, 560, 640'
            },
            "Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)": {
                "Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550],
                "Baseline load": [5.4, 5.0, 4.9, 6.4],
                "Spatial carbon": '1: 540, 500, 320, 600; 2: 320, 510, 540, 600; 3: 560, 330, 520, 610; 4: 620, 560, 500, 330; 5: 330, 420, 550, 640'
            },
            "Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)": {
                "Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535],
                "Baseline load": [5.0, 5.1, 5.0, 6.7],
                "Spatial carbon": '1: 320, 520, 560, 600; 2: 550, 330, 520, 580; 3: 600, 540, 500, 320; 4: 560, 500, 330, 540; 5: 500, 340, 560, 630'
            },
            "Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)": {
                "Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545],
                "Baseline load": [5.2, 5.3, 5.0, 6.6],
                "Spatial carbon": '1: 510, 330, 560, 600; 2: 560, 500, 320, 590; 3: 320, 520, 540, 620; 4: 630, 560, 510, 340; 5: 330, 420, 560, 630'
            },
            "Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)": {
                "Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540],
                "Baseline load": [5.5, 5.2, 4.8, 6.5],
                "Spatial carbon": '1: 540, 500, 320, 610; 2: 320, 510, 560, 620; 3: 560, 340, 520, 610; 4: 640, 560, 510, 330; 5: 520, 330, 540, 600'
            },
            "Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)": {
                "Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530],
                "Baseline load": [5.1, 4.9, 4.8, 6.3],
                "Spatial carbon": '1: 330, 520, 560, 610; 2: 540, 330, 520, 600; 3: 580, 540, 330, 620; 4: 630, 560, 500, 330; 5: 520, 330, 550, 600'
            }
        }
    }

    # Agent 3 Profile
    AGENT_PROFILE = {
        'persona': 'Position 3 night-shift nurse on the central ridge',
        'location': 3,
        'base_demand': [0.60, 0.80, 0.90, 0.70]
    }

    NEIGHBORS = {
        'Neighbor 2': {'location': 2, 'Base demand': [0.70, 1.00, 0.80, 0.50], 'Preferred slots': [1, 2], 'Comfort penalty': 0.14},
        'Neighbor 5': {'location': 5, 'Base demand': [0.50, 0.70, 0.60, 0.90], 'Preferred slots': [0, 1], 'Comfort penalty': 0.12}
    }

    policy_engine = AgentPolicy(
        persona=AGENT_PROFILE['persona'],
        location=AGENT_PROFILE['location'],
        base_demand=AGENT_PROFILE['base_demand'],
        neighbors=NEIGHBORS,
        forecast=SCENARIO_CONTEXT
    )
    
    plan = policy_engine.generate_plan()
    
    # 3. Write local_policy_output.json
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(plan, f, indent=4)

if __name__ == "__main__":
    main()