import json
import os

class AgentPolicy:
    def __init__(self, scenario_data, agent_id, agent_location):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.location = agent_location
        self.slots = self.scenario['slots']
        self.num_slots = len(self.slots)
        self.base_demand = self._get_base_demand()
        self.slot_limits = self._get_slot_limits()
        
        # Heuristic adjustments based on nudge:
        # Original preference: Late charging (Slot 3 preferred or Slot 2/3 based on cost).
        # Nudge suggests shifting from Slot 3 to Slot 2 on specific days (2, 4, 6) for coordination benefit.
        # Since the nudge explicitly mentions *my* preference aligns with Slot 3, but suggests moving 
        # to Slot 2 on specific days to benefit coordination (which I do not directly care about, 
        # but the benefit of "cutting peak contribution by nearly 20%" might imply a future benefit 
        # like avoiding future restrictions or smoothing my own demand profile, thus a minor benefit), 
        # I will slightly adjust my inherent preference towards Slot 2 on those specific days 
        # if the cost difference is small, or if the cost analysis already leans that way.
        
        # Since the current cost function only minimizes *personal* cost (Price + Carbon), 
        # I must check if the cost in Slot 2 on those days is comparable to Slot 3.
        
        # Define days where coordination benefit is highlighted (Day indices 1, 3, 5 corresponding to Days 2, 4, 6)
        self.coordination_days = {1, 3, 5} 
        
        # No direct measurable change to the cost function parameters (K, base_demand) unless 
        # the nudge implied a change in my personal goals. It implies a coordination benefit.
        # I will maintain the personal cost minimization but observe if the resulting plan shifts 
        # towards Slot 2 on coordination days due to the inherent costs making Slot 2 nearly as cheap.
        # If Slot 3 remains strictly cheaper after cost calculation, I stick to Slot 3 as per Stage 2 rules.
        # If the nudge suggests a benefit (20% peak reduction), I will slightly favor Slot 2 
        # *if* the cost is within a small margin (e.g., 5% difference).

        self.cost_tolerance_factor = 1.05 # If Slot 2 cost is <= 1.05 * Slot 3 cost on coordination days, pick Slot 2.

    def _get_base_demand(self):
        # Base demand for Agent 5 (index 4 in the scenario list)
        return [0.50, 0.70, 0.60, 0.90]

    def _get_slot_limits(self):
        return {
            'min': [self.scenario['slot_min_sessions'][str(i)] for i in range(self.num_slots)],
            'max': [self.scenario['slot_max_sessions'][str(i)] for i in range(self.num_slots)]
        }

    def _get_spatial_carbon(self, day_data, location):
        spatial_str = day_data.get('Spatial carbon')
        if not spatial_str:
            raise ValueError(f"Could not find Spatial Carbon data for {day_data.keys()}")

        for loc_data in spatial_str.split(';'):
            if loc_data.strip().startswith(f"{location}:"):
                carbon_values = loc_data.strip().split(':')[1].strip().split(',')
                return [int(c.strip()) for c in carbon_values]
        
        raise ValueError(f"Spatial carbon data for location {location} not found in day block.")


    def calculate_cost(self, day_key, slot_index):
        day_data = self.scenario['days'][day_key]
        
        tariff = day_data['Tariff'][slot_index]
        carbon = day_data['Carbon'][slot_index]
        
        try:
            spatial_carbon_list = self._get_spatial_carbon(day_data, self.location)
            spatial_carbon = spatial_carbon_list[slot_index]
        except ValueError:
            spatial_carbon = carbon

        demand_factor = self.base_demand[slot_index]
        
        K = 0.0005
        cost = tariff + (carbon * K) 
        
        return cost, demand_factor

    def plan_schedule(self):
        day_keys = list(self.scenario['days'].keys())
        plan = []

        for day_index, day_key in enumerate(day_keys):
            best_cost = float('inf')
            best_slot = -1
            
            # First pass: Calculate all costs
            costs = {}
            for i in range(self.num_slots):
                cost, _ = self.calculate_cost(day_key, i)
                costs[i] = cost
            
            # Determine the objectively cheapest slot based on personal cost
            cheapest_cost = min(costs.values())
            cheapest_slots = [slot for slot, cost in costs.items() if cost == cheapest_cost]
            
            # Agent 5 is a late charger, prefers slots 2 or 3 if costs are equal.
            # Default tie-breaker: Pick the latest among the cheapest slots.
            if cheapest_slots:
                best_slot = max(cheapest_slots)
            else:
                best_slot = 3 # Default fallback to original late preference if calculation failed oddly

            
            # --- NUDGE APPLICATION ---
            # If we are on a coordination day (Days 2, 4, 6 -> indices 1, 3, 5)
            # and the preferred slot (Slot 3) is the cheapest OR tied for cheapest, 
            # check if Slot 2 is acceptably close in cost (within 5% tolerance).
            
            if day_index in self.coordination_days:
                slot_3_cost = costs.get(3, float('inf'))
                slot_2_cost = costs.get(2, float('inf'))

                # Nudge suggests favoring Slot 2 over Slot 3 if it eases pressure.
                # If Slot 3 was selected (because it was the cheapest/tied)
                if best_slot == 3:
                    # Check if Slot 2 is close enough to Slot 3's cost.
                    if slot_2_cost <= slot_3_cost * self.cost_tolerance_factor:
                        # If Slot 2 is nearly as cheap, adopt the suggested coordination shift.
                        best_slot = 2
                
                # Edge case: If Slot 2 was already the cheapest (best_slot == 2), we keep it.
                # Edge case: If Slot 2 was not the cheapest, we only move from 3 to 2.
                
            plan.append(best_slot)
            
        return plan

def main():
    # --- Load context from prompt ---
    slots_data = {
        0: "19-20", 1: "20-21", 2: "21-22", 3: "22-23"
    }
    
    scenario_context = {
        "scenario_id": "ev_peak_sharing_1",
        "slots": slots_data,
        "price": [0.23, 0.24, 0.27, 0.30],
        "carbon_intensity": [700, 480, 500, 750],
        "capacity": 6.8,
        "baseline_load": [5.2, 5.0, 4.9, 6.5],
        "slot_min_sessions": {str(i): 1 for i in range(4)},
        "slot_max_sessions": {str(i): 2 if i != 2 else 1 for i in range(4)},
        "spatial_carbon": "1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | 3: 503, 473, 471, 577 | 4: 617, 549, 479, 363 | 5: 411, 376, 554, 623",
        "days": {
            "Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)": {
                "Tariff": [0.20, 0.25, 0.29, 0.32],
                "Carbon": [490, 470, 495, 540],
                "Baseline load": [5.3, 5.0, 4.8, 6.5],
                "Spatial carbon": "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; 3: 590, 520, 340, 630; 4: 620, 560, 500, 330; 5: 360, 380, 560, 620"
            },
            "Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)": {
                "Tariff": [0.27, 0.22, 0.24, 0.31],
                "Carbon": [485, 460, 500, 545],
                "Baseline load": [5.1, 5.2, 4.9, 6.6],
                "Spatial carbon": "1: 510, 330, 550, 600; 2: 540, 500, 320, 610; 3: 310, 520, 550, 630; 4: 620, 540, 500, 340; 5: 320, 410, 560, 640"
            },
            "Day 3 (Day 3 — Marine layer shifts low-carbon pocket to the early slots.)": {
                "Tariff": [0.24, 0.21, 0.26, 0.30],
                "Carbon": [500, 455, 505, 550],
                "Baseline load": [5.4, 5.0, 4.9, 6.4],
                "Spatial carbon": "1: 540, 500, 320, 600; 2: 320, 510, 540, 600; 3: 560, 330, 520, 610; 4: 620, 560, 500, 330; 5: 330, 420, 550, 640"
            },
            "Day 4 (Day 4 — Neighborhood watch enforces staggered use before the late-event recharge.)": {
                "Tariff": [0.19, 0.24, 0.28, 0.22],
                "Carbon": [495, 470, 500, 535],
                "Baseline load": [5.0, 5.1, 5.0, 6.7],
                "Spatial carbon": "1: 320, 520, 560, 600; 2: 550, 330, 520, 580; 3: 600, 540, 500, 320; 4: 560, 500, 330, 540; 5: 500, 340, 560, 630"
            },
            "Day 5 (Day 5 — Festival lighting brings high-carbon spikes after 22h.)": {
                "Tariff": [0.23, 0.20, 0.27, 0.31],
                "Carbon": [500, 450, 505, 545],
                "Baseline load": [5.2, 5.3, 5.0, 6.6],
                "Spatial carbon": "1: 510, 330, 560, 600; 2: 560, 500, 320, 590; 3: 320, 520, 540, 620; 4: 630, 560, 510, 340; 5: 330, 420, 560, 630"
            },
            "Day 6 (Day 6 — Maintenance advisory caps the valley transformer; slot 2 is rationed.)": {
                "Tariff": [0.26, 0.22, 0.25, 0.29],
                "Carbon": [505, 460, 495, 540],
                "Baseline load": [5.5, 5.2, 4.8, 6.5],
                "Spatial carbon": "1: 540, 500, 320, 610; 2: 320, 510, 560, 620; 3: 560, 340, 520, 610; 4: 640, 560, 510, 330; 5: 520, 330, 540, 600"
            },
            "Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)": {
                "Tariff": [0.21, 0.23, 0.28, 0.26],
                "Carbon": [495, 460, 500, 530],
                "Baseline load": [5.1, 4.9, 4.8, 6.3],
                "Spatial carbon": "1: 330, 520, 560, 610; 2: 540, 330, 520, 600; 3: 580, 540, 330, 620; 4: 630, 560, 500, 330; 5: 520, 330, 550, 600"
            }
        }
    }
    
    agent = AgentPolicy(
        scenario_data=scenario_context, 
        agent_id=5, 
        agent_location=5
    )
    
    schedule = agent.plan_schedule()
    
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(schedule, f)

if __name__ == "__main__":
    main()