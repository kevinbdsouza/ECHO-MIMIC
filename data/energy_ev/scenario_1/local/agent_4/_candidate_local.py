import json

class AgentPolicy:
    def __init__(self):
        # Agent 4: Position 4 retirees guarding comfort and grid warnings
        self.location = 4
        # Base demand for slots 0, 1, 2, 3. Slot 0 is highest comfort priority (0.90).
        self.base_demand = [0.90, 0.60, 0.70, 0.80] 

        # Ground truth choices from neighbors used for imitation context (Slot 1 is dominant)
        # N3 GT: [1, 1, 1, 3, 1, 1, 1]
        # N5 GT: [0, 1, 0, 0, 1, 1, 1]
        self.neighbor_gt_5 = {
            'Day 1': 0, 'Day 2': 1, 'Day 3': 0, 'Day 4': 0, 'Day 5': 1, 'Day 6': 1, 'Day 7': 1
        }
        
        # Determine the primary imitation target based on neighbor consensus
        self.imitation_targets = {}
        for i in range(1, 8):
            day_key = f'Day {i}'
            # Neighbor 3 is always slot 1 except Day 4
            n3_gt = 1 if day_key != 'Day 4' else 3 
            n5_gt = self.neighbor_gt_5[day_key]
            
            if n3_gt == n5_gt:
                self.imitation_targets[day_key] = n3_gt
            else:
                # Divergence. Default to slot 1 as the most frequent choice among neighbors overall.
                self.imitation_targets[day_key] = 1 


    def calculate_cost(self, day_data, slot_idx, capacity):
        price = day_data['Tariff'][slot_idx]
        carbon = day_data['Carbon'][slot_idx]
        baseline = day_data['Baseline load'][slot_idx]
        
        # Spatial Carbon for Location 4 (index 3 in spatial carbon lists provided in daily context)
        spatial_carbon_str = day_data['Spatial carbon'][self.location - 1]
        spatial_carbon_values = [int(x) for x in spatial_carbon_str.split(', ')]
        spatial_carbon = spatial_carbon_values[slot_idx]

        # Capacity Warning 
        capacity_factor = 0.0
        if baseline + 1.0 > capacity: 
            capacity_factor = 1.5 * (baseline + 1.0 - capacity) 

        # Comfort/Demand Cost: High base demand in Slot 0
        demand_penalty = self.base_demand[slot_idx] * 0.1 
        
        # Cost Function: Price + Carbon (weighted high for grid warnings) + Comfort + Capacity
        W_C = 0.0015 
        
        total_cost = price + W_C * carbon + demand_penalty + capacity_factor
        
        # Apply hard constraint penalty (Day 6 rationing for Slot 2)
        if slot_idx == 2 and 'Day 6' in day_data['Day Name']:
            total_cost += 10.0 

        return total_cost

    def get_slot_constraints(self, day_name, slot_idx):
        if slot_idx == 2 and 'Day 6' in day_name:
            return 0, 0 # Rationed
        
        return scenario_context['slot_min_sessions'][slot_idx], scenario_context['slot_max_sessions'][slot_idx]

    def choose_slot(self, day_name, day_data):
        
        evaluated_costs = {}
        min_cost = float('inf')
        best_slot_by_cost = -1
        
        # 1. Calculate costs and determine the mathematically cheapest slot for Agent 4
        for i in range(4):
            min_s, max_s = self.get_slot_constraints(day_name, i)
            
            if min_s == 0 and max_s == 0:
                cost = float('inf')
            else:
                cost = self.calculate_cost(day_data, i, scenario_context['capacity'])
            
            evaluated_costs[i] = cost
            
            if cost < min_cost:
                min_cost = cost
                best_slot_by_cost = i

        # 2. Apply Imitation Override
        imitation_target = self.imitation_targets[day_name]
        target_cost = evaluated_costs.get(imitation_target, float('inf'))
        
        # If the imitation target is available AND its cost is within 50% of the absolute minimum cost found, choose the target slot.
        if target_cost < float('inf') and (target_cost <= min_cost * 1.5 or min_cost == float('inf')):
            return imitation_target
            
        # Otherwise, choose the slot that minimizes Agent 4's calculated cost
        return best_slot_by_cost

# --- Context Loading Simulation ---
scenario_context = {
    'slots': {0: '19-20', 1: '20-21', 2: '21-22', 3: '22-23'},
    'price': [0.23, 0.24, 0.27, 0.30],
    'carbon_intensity': [700, 480, 500, 750],
    'capacity': 6.8,
    'baseline_load': [5.2, 5.0, 4.9, 6.5],
    'slot_min_sessions': {0: 1, 1: 1, 2: 1, 3: 1},
    'slot_max_sessions': {0: 2, 1: 2, 2: 1, 3: 2},
    'spatial_carbon': {
        1: '440, 460, 490, 604', 2: '483, 431, 471, 600', 3: '503, 473, 471, 577',
        4: '617, 549, 479, 363', 5: '411, 376, 554, 623'
    }
}

daily_data = {
    'Day 1': {'Day Name': 'Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)', 'Tariff': [0.20, 0.25, 0.29, 0.32], 'Carbon': [490, 470, 495, 540], 'Baseline load': [5.3, 5.0, 4.8, 6.5], 'Spatial carbon': {1: '330, 520, 560, 610', 2: '550, 340, 520, 600', 3: '590, 520, 340, 630', 4: '620, 560, 500, 330', 5: '360, 380, 560, 620'}},
    'Day 2': {'Day Name': 'Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)', 'Tariff': [0.27, 0.22, 0.24, 0.31], 'Carbon': [485, 460, 500, 545], 'Baseline load': [5.1, 5.2, 4.9, 6.6], 'Spatial carbon': {1: '510, 330, 550, 600', 2: '540, 500, 320, 610', 3: '310, 520, 550, 630', 4: '620, 540, 500, 340', 5: '320, 410, 560, 640'}},
    'Day 3': {'Day Name': 'Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)', 'Tariff': [0.24, 0.21, 0.26, 0.30], 'Carbon': [500, 455, 505, 550], 'Baseline load': [5.4, 5.0, 4.9, 6.4], 'Spatial carbon': {1: '540, 500, 320, 600', 2: '320, 510, 540, 600', 3: '560, 330, 520, 610', 4: '620, 560, 500, 330', 5: '330, 420, 550, 640'}},
    'Day 4': {'Day Name': 'Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)', 'Tariff': [0.19, 0.24, 0.28, 0.22], 'Carbon': [495, 470, 500, 535], 'Baseline load': [5.0, 5.1, 5.0, 6.7], 'Spatial carbon': {1: '320, 520, 560, 600', 2: '550, 330, 520, 580', 3: '600, 540, 500, 320', 4: '560, 500, 330, 540', 5: '500, 340, 560, 630'}},
    'Day 5': {'Day Name': 'Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)', 'Tariff': [0.23, 0.20, 0.27, 0.31], 'Carbon': [500, 450, 505, 545], 'Baseline load': [5.2, 5.3, 5.0, 6.6], 'Spatial carbon': {1: '510, 330, 560, 600', 2: '560, 500, 320, 590', 3: '320, 520, 540, 620', 4: '630, 560, 510, 340', 5: '330, 420, 560, 630'}},
    'Day 6': {'Day Name': 'Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)', 'Tariff': [0.26, 0.22, 0.25, 0.29], 'Carbon': [505, 460, 495, 540], 'Baseline load': [5.5, 5.2, 4.8, 6.5], 'Spatial carbon': {1: '540, 500, 320, 610', 2: '320, 510, 560, 620', 3: '560, 340, 520, 610', 4: '640, 560, 510, 330', 5: '520, 330, 540, 600'}},
    'Day 7': {'Day Name': 'Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)', 'Tariff': [0.21, 0.23, 0.28, 0.26], 'Carbon': [495, 460, 500, 530], 'Baseline load': [5.1, 4.9, 4.8, 6.3], 'Spatial carbon': {1: '330, 520, 560, 610', 2: '540, 330, 520, 600', 3: '580, 540, 330, 620', 4: '630, 560, 500, 330', 5: '520, 330, 550, 600'}},
}

day_names_ordered = [
    'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'
]

# --- Execution ---
policy = AgentPolicy()
slot_plan = []

for day_key in day_names_ordered:
    day_data = daily_data[day_key]
    chosen_slot = policy.choose_slot(day_key, day_data)
    slot_plan.append(chosen_slot)

# --- Output Generation ---
output_filename = "local_policy_output.json"
with open(output_filename, 'w') as f:
    json.dump(slot_plan, f)