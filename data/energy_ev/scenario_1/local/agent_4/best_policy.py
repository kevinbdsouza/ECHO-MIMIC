import json
import os

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbor_examples, scenario_context):
        self.persona = persona
        self.location = location
        self.base_demand = base_demand
        self.neighbor_examples = neighbor_examples
        self.scenario_context = scenario_context
        
        # Agent 4: Position 4 retirees guarding comfort and grid warnings.
        # Prioritize comfort (high base demand in early slots 0, 1) and 
        # be wary of grid warnings (high carbon/price).
        # Base demand: 0.90, 0.60, 0.70, 0.80 -> Heaviest in Slot 0 (19-20h)
        self.slot_weights = [1.5, 0.8, 0.7, 1.0] # Higher weight for comfort (Slot 0)

    def calculate_cost(self, day_data, slot_index):
        tariff = day_data['Tariff'][slot_index]
        carbon = day_data['Carbon'][slot_index]
        
        # Spatial carbon for location 4
        spatial_carbon_str = day_data['Spatial carbon'][self.location - 1]
        spatial_carbon_values = [int(x) for x in spatial_carbon_str.split(', ')]
        spatial_carbon = spatial_carbon_values[slot_index]
        
        comfort_factor = self.slot_weights[slot_index]
        
        # Cost components:
        # Agent 4 values comfort highly (Slot 0 preferred due to high weight/low inverse weight).
        # Carbon is secondary concern ("grid warnings").
        
        cost = (
            tariff * 1.0 +              # Price (Primary cost)
            carbon * 0.001 +            # Carbon (Secondary concern)
            (1 / comfort_factor) * 0.5  # Comfort: Lower cost means higher preference (Slot 0 minimizes this term)
        )
        
        return cost

    def choose_slots(self):
        days_data = self.scenario_context['days']
        all_slots = list(range(len(self.scenario_context['price'])))
        
        chosen_slots = []
        
        for day_name in days_data:
            day_data = days_data[day_name]
            
            best_slot = -1
            min_cost = float('inf')
            
            slot_maxs = self.scenario_context['slot_max_sessions']

            possible_slots = []
            for i in all_slots:
                # Check if participation is allowed (max sessions must be >= 1)
                if slot_maxs[i] >= 1:
                    possible_slots.append(i)

            
            for slot_index in possible_slots:
                cost = self.calculate_cost(day_data, slot_index)
                
                if cost < min_cost:
                    min_cost = cost
                    best_slot = slot_index
            
            # Safety check for specific Day 6 constraint mentioned in prompt for Slot 2 rationing:
            if "Day 6" in day_name and best_slot == 2:
                 # If slot 2 was cheapest but is rationed, prioritize Slot 0 (highest comfort preference) 
                 # if its cost is within a reasonable tolerance (20%) of the actual minimum cost found.
                 cost_slot_0 = self.calculate_cost(day_data, 0)
                 if cost_slot_0 < min_cost * 1.2: 
                     best_slot = 0
            
            if best_slot == -1:
                # Fallback: choose Slot 0 (highest comfort priority)
                best_slot = 0 
            
            chosen_slots.append(best_slot)
            
        return chosen_slots

if __name__ == '__main__':
    # --- Scenario Context Definition ---
    scenario_context = {
        'slots': {0: '19-20', 1: '20-21', 2: '21-22', 3: '22-23'},
        'price': [0.23, 0.24, 0.27, 0.30],
        'carbon_intensity': [700, 480, 500, 750],
        'capacity': 6.8,
        'baseline_load': [5.2, 5.0, 4.9, 6.5],
        'slot_min_sessions': {0: 1, 1: 1, 2: 1, 3: 1},
        'slot_max_sessions': {0: 2, 1: 2, 2: 1, 3: 2},
        'spatial_carbon': {
            1: '440, 460, 490, 604', 
            2: '483, 431, 471, 600', 
            3: '503, 473, 471, 577', 
            4: '617, 549, 479, 363', 
            5: '411, 376, 554, 623'
        },
        'days': {
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
    
    # Agent 4 profile
    persona = "Position 4 retirees guarding comfort and grid warnings"
    location = 4
    base_demand = [0.90, 0.60, 0.70, 0.80]
    neighbor_examples = [
        {'position': 3, 'base_demand': [0.60, 0.80, 0.90, 0.70], 'preferred_slots': [1, 3], 'comfort_penalty': 0.20},
        {'position': 5, 'base_demand': [0.50, 0.70, 0.60, 0.90], 'preferred_slots': [0, 1], 'comfort_penalty': 0.12}
    ]

    agent = AgentPolicy(persona, location, base_demand, neighbor_examples, scenario_context)
    daily_slot_plan = agent.choose_slots()

    # Output to local_policy_output.json
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(daily_slot_plan, f, indent=4)