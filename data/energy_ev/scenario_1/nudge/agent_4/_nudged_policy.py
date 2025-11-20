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

        # Initialize heuristic based on persona evaluation.
        # The nudge suggests shifting usage on Day 3 and Day 6 to Slot 1 from Slot 0,
        # which benefits the 'grid warnings' concern (spatial carbon reduction).
        # Since Slot 0 remains high priority (comfort), we will only adjust if the benefit is clear,
        # and only slightly, as requested ("make only measurable changes to your behaviour and not drastic ones").
        # Since the nudge explicitly targets Days 3 and 6 for a shift to Slot 1 (from likely Slot 0), 
        # we will incorporate this by slightly reducing the weight/preference for Slot 0 on those specific days,
        # or by implementing the shift if the cost calculation still favors it, but we must check the *current* output first.
        
        # Current policy: calculate_cost will determine slots. We only adjust slot_weights if the nudge 
        # strongly suggests changing baseline behavior. The nudge suggests a specific beneficial coordination:
        # Move from Slot 0 to Slot 1 on Day 3 and Day 6.
        
        # We will modify slot_weights based on the suggestion, but only marginally, 
        # focusing the change on the specific days mentioned if possible, otherwise globally.
        # Since the current implementation selects slots based on cost minimization across all days,
        # a global change to weights is the only way to implement a suggested preference change.
        
        # Original weights: [1.5, 0.8, 0.7, 1.0] -> Inverse comfort cost: [0.66, 1.25, 1.42, 1.0]
        # Slot 0 is heavily favored.
        
        # Nudge suggests moving some load from Slot 0 to Slot 1 on D3, D6 for carbon benefits.
        # Slot 1 is already moderately preferred (0.8). Slot 0 is the comfort core.
        # To reflect a *slight* willingness to shift load from Slot 0 to Slot 1 *if carbon is favorable*, 
        # we will slightly decrease the comfort weight of Slot 0 and slightly increase Slot 1's weight, 
        # making Slot 1 more competitive against Slot 0 when price/carbon overlap favorably.
        
        self.slot_weights = [1.45, 0.85, 0.7, 1.0] # Reduced Slot 0 weight slightly (1.5 -> 1.45), increased Slot 1 (0.8 -> 0.85)

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
        
        # To implement the targeted shifts (Day 3, Day 6 to Slot 1) explicitly, 
        # we check the day name during selection and apply a temporary override/strong preference adjustment 
        # if the cost logic *without* the override still favors Slot 0 too strongly, reflecting the nudge's coordination intent.
        
        day_names = list(days_data.keys())
        
        for i, day_name in enumerate(day_names):
            day_data = days_data[day_name]
            
            best_slot = -1
            min_cost = float('inf')
            
            slot_maxs = self.scenario_context['slot_max_sessions']

            possible_slots = []
            for j in all_slots:
                # Check if participation is allowed (max sessions must be >= 1)
                if slot_maxs[j] >= 1:
                    possible_slots.append(j)

            
            for slot_index in possible_slots:
                cost = self.calculate_cost(day_data, slot_index)
                
                # Apply specific coordination adjustment based on nudge for D3 and D6
                if "Day 3" in day_name or "Day 6" in day_name:
                    # Nudge suggested shifting load from Slot 0 to Slot 1. 
                    # If Slot 1 is the second-best option, boost its attractiveness temporarily to ensure it wins over Slot 0 
                    # if Slot 0 is only marginally cheaper (e.g., within 10% cost margin).
                    
                    current_cost = cost
                    
                    if slot_index == 1:
                        # Increase incentive for Slot 1 on D3/D6 to encourage coordination shift
                        # Penalty for Slot 0 (comfort driver) is already high. If Slot 1 is close, pull it ahead.
                        if min_cost != float('inf'):
                            cost_if_slot_0_wins = self.calculate_cost(day_data, best_slot) if best_slot != -1 else float('inf')
                            
                            if best_slot == 0 and current_cost < cost_if_slot_0_wins * 1.10:
                                # If Slot 0 is currently winning, and Slot 1 is close, we artificially lower Slot 1's cost
                                # to ensure the coordination shift happens, reflecting the external benefit guidance.
                                pass # We handle this by checking against the current minimum below.

                    if slot_index == 0:
                         # Slightly penalize Slot 0 on D3/D6, but only if Slot 1 is available and participating
                         if 1 in possible_slots and self.calculate_cost(day_data, 1) < float('inf'):
                            # If Slot 1 is a viable option, we reduce Slot 0's competitiveness slightly here 
                            # to reflect the coordination effort suggested by the nudge.
                            # We subtract a small amount from Slot 0's calculated cost if it's the current minimum, 
                            # but since we iterate, we just rely on the comparison logic below.
                            pass
                            
                    cost = current_cost # Keep cost calculation standard, rely on comparison logic for selection bias.

                
                if cost < min_cost:
                    min_cost = cost
                    best_slot = slot_index
                elif cost == min_cost:
                    # Tie-breaker: If costs are equal, maintain current preference unless the nudge suggests otherwise.
                    # Nudge suggests favoring Slot 1 over Slot 0 on D3/D6.
                    if ("Day 3" in day_name or "Day 6" in day_name) and slot_index == 1 and best_slot == 0:
                        best_slot = 1 # Prefer Slot 1 in a tie on D3/D6
                    elif slot_index == 0 and best_slot != 0:
                        # Default tie-breaker favors Slot 0 due to higher comfort weight
                        best_slot = 0 # This should usually be true already unless another slot was already selected as best
                        

            # --- Post-iteration logic ---

            # If D3 or D6, and Slot 0 won, check if Slot 1 was close and enforce the shift if it benefits carbon (as suggested by nudge)
            if ("Day 3" in day_name or "Day 6" in day_name) and best_slot == 0:
                 cost_slot_0 = self.calculate_cost(day_data, 0)
                 if 1 in possible_slots:
                    cost_slot_1 = self.calculate_cost(day_data, 1)
                    
                    # If Slot 1 is within 10% of Slot 0's cost, and Slot 1 has lower spatial carbon (the explicit benefit mentioned)
                    spatial_carbon_str = day_data['Spatial carbon'][self.location - 1]
                    spatial_carbon_values = [int(x) for x in spatial_carbon_str.split(', ')]
                    
                    if cost_slot_1 < cost_slot_0 * 1.10 and spatial_carbon_values[1] < spatial_carbon_values[0]:
                        best_slot = 1 # Enforce shift to Slot 1 based on nudge coordination goal

            
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
            
        # The scenario only defines 7 days, but only 4 slots. We need exactly 7 outputs.
        # Ensure we have 7 outputs if the input dictionary structure implies 7 distinct scheduling days.
        # Since the original code used a fixed loop over days_data, and we iterate over all keys in days_data, 
        # we should get exactly 7 results if there are 7 days defined.

        # If the loop resulted in fewer than 7 days (e.g., if the environment truncated days_data), we pad/handle.
        # Given the setup, we assume 7 days are present and iterated over.
        
        # Final check: If the final list size is not 7 (which is required by the output specification), 
        # pad/truncate based on the number of days processed.
        while len(chosen_slots) < 7:
            chosen_slots.append(0) # Default to slot 0 if we somehow missed a day in iteration
        return chosen_slots[:7]


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