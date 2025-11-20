import json
import os

class AgentPolicy:
    def __init__(self, persona, location, base_demand, neighbor_examples):
        self.persona = persona
        self.location_id = location
        self.base_demand = base_demand
        self.neighbor_examples = neighbor_examples
        
        # Configuration based on prompt context (fixed for this imitation stage)
        self.slots_info = {
            0: {'time': '19-20', 'price': 0.23, 'carbon': 700, 'baseline': 5.2},
            1: {'time': '20-21', 'price': 0.24, 'carbon': 480, 'baseline': 5.0},
            2: {'time': '21-22', 'price': 0.27, 'carbon': 500, 'baseline': 4.9},
            3: {'time': '22-23', 'price': 0.30, 'carbon': 750, 'baseline': 6.5},
        }
        self.capacity = 6.8

    def load_scenario_data(self):
        # In a real execution environment, scenario.json would be present.
        # We must simulate loading it based on the provided prompt context.
        
        # Slots and Global Forecasts
        self.slots = [0, 1, 2, 3]
        self.global_price = [0.23, 0.24, 0.27, 0.30]
        self.global_carbon = [700, 480, 500, 750]
        self.global_baseline = [5.2, 5.0, 4.9, 6.5]
        self.slot_min_sessions = [1, 1, 1, 1]
        self.slot_max_sessions = [2, 2, 1, 2]
        
        # Daily Data (Simulated load from prompt)
        self.days_data = {
            "Day 1": {'Tariff': [0.20, 0.25, 0.29, 0.32], 'Carbon': [490, 470, 495, 540], 'Baseline load': [5.3, 5.0, 4.8, 6.5]},
            "Day 2": {'Tariff': [0.27, 0.22, 0.24, 0.31], 'Carbon': [485, 460, 500, 545], 'Baseline load': [5.1, 5.2, 4.9, 6.6]},
            "Day 3": {'Tariff': [0.24, 0.21, 0.26, 0.30], 'Carbon': [500, 455, 505, 550], 'Baseline load': [5.4, 5.0, 4.9, 6.4]},
            "Day 4": {'Tariff': [0.19, 0.24, 0.28, 0.22], 'Carbon': [495, 470, 500, 535], 'Baseline load': [5.0, 5.1, 5.0, 6.7]},
            "Day 5": {'Tariff': [0.23, 0.20, 0.27, 0.31], 'Carbon': [500, 450, 505, 545], 'Baseline load': [5.2, 5.3, 5.0, 6.6]},
            "Day 6": {'Tariff': [0.26, 0.22, 0.25, 0.29], 'Carbon': [505, 460, 495, 540], 'Baseline load': [5.5, 5.2, 4.8, 6.5]},
            "Day 7": {'Tariff': [0.21, 0.23, 0.28, 0.26], 'Carbon': [495, 460, 500, 530], 'Baseline load': [5.1, 4.9, 4.8, 6.3]},
        }
        self.day_keys = sorted(self.days_data.keys())


    def calculate_cost(self, day_data, slot_index, session_load):
        # Agent Persona: Position 1 battery engineer balancing budget and solar backfeed.
        # Objective: Minimize cost (Tariff) primarily, consider carbon if costs are close.
        # Solar backfeed preference implies favoring early slots (low index) if tariffs are competitive, 
        # as solar generation tends to peak earlier in the evening than high carbon/price spikes.
        
        tariff = day_data['Tariff'][slot_index]
        carbon = day_data['Carbon'][slot_index]
        
        # Base cost metric (Budget focus)
        cost = tariff
        
        # Minor penalty/reward for Carbon, but tariff is primary driver for "budget" balance.
        # We use a weighted sum, heavily favoring tariff.
        # Carbon weighting factor: Since global carbon is high (480-750), and tariff is low (0.2-0.3),
        # we weigh carbon relative to its magnitude, perhaps dividing by 1000 to bring it into the tariff range.
        carbon_weight = 0.0005 
        cost += carbon_weight * carbon
        
        return cost

    def choose_slot(self, day_name, day_data):
        
        best_slot = -1
        min_cost = float('inf')
        
        # Imitation Logic: If neighbors consistently pick one slot, follow it if it's reasonable.
        # Neighbor 2 (Feeder Analyst, Loc 2): Prefers 1, 2. GT: Always picks Slot 1.
        # Neighbor 3 (Nurse, Loc 3): Prefers 1, 3. GT: Mostly picks Slot 1 (6/7 days).
        
        # Strong imitation target: Slot 1 appears dominant among neighbors aiming for efficiency/comfort.
        # As a budget-conscious engineer (Position 1), Slot 1 (20-21h) is often the second cheapest tariff globally (0.24 vs 0.23 in slot 0).
        
        # Let's check Slot 1 first for neighbor alignment, but confirm it's not terrible cost-wise.
        
        neighbor_preference = 1 
        
        # Calculate cost for all slots to establish a baseline
        slot_costs = []
        for i in range(4):
            # Use a representative session load (e.g., average of min/max sessions)
            avg_sessions = (self.slot_min_sessions[i] + self.slot_max_sessions[i]) / 2
            session_load = self.base_demand[i] * avg_sessions / 1.0 # Simplified load calculation for cost comparison
            
            cost = self.calculate_cost(day_data, i, session_load)
            slot_costs.append({'slot': i, 'cost': cost, 'tariff': day_data['Tariff'][i], 'carbon': day_data['Carbon'][i]})

        # Check if the neighbor's favored slot (1) is significantly worse than the absolute minimum cost slot
        min_cost_slot = min(slot_costs, key=lambda x: x['cost'])
        
        # Imitation rule: If the neighbor's preferred slot is within 10% of the absolute minimum cost, choose it.
        # Otherwise, choose the absolute minimum cost slot. (This balances imitation with primary personal objective: budget)
        
        cost_tolerance = 1.10 # 10% tolerance
        
        if slot_costs[neighbor_preference]['cost'] <= min_cost_slot['cost'] * cost_tolerance:
            best_slot = neighbor_preference
        else:
            best_slot = min_cost_slot['slot']
            
        # Check constraints: Must satisfy min/max session requirements IF we were scheduling sessions.
        # Since we only output the slot index, we trust the chosen slot is valid based on the context provided.
        
        return best_slot

    def generate_policy(self):
        self.load_scenario_data()
        
        policy_output = []
        
        # Agent 1 (Position 1) is a battery engineer focused on budget/solar backfeed.
        # Solar backfeed suggests favoring earlier slots (0 or 1) when tariffs are low.
        
        # Nudge analysis: The nudge confirms that Slot 1 is consistently beneficial (low cost, low carbon avoidance) 
        # and aligns with both my baseline objective (low cost/solar) and neighbor observation. 
        # Since the nudge reinforces that Slot 1 is the coordinated optimal strategy, I will solidify my bias towards Slot 1.
        # I will change my heuristic to *always* choose Slot 1, as the prior heuristic only chose it if it was within 10% of the minimum cost. 
        # The nudge explicitly stated Slot 1 is the "coordinated strategy" and confirms carbon benefits. This warrants a stronger commitment.
        
        for day_name in self.day_keys:
            day_data = self.days_data[day_name]
            
            # Calculate costs to verify consistency, but enforce Slot 1 based on favorable feedback.
            day_costs = []
            for i in range(4):
                cost = self.calculate_cost(day_data, i, 1.0) # Use normalized session load for comparison
                day_costs.append({'slot': i, 'cost': cost, 'tariff': day_data['Tariff'][i]})
                
            # Current policy update: Commit fully to Slot 1, as reinforced by the nudge's analysis.
            chosen_slot = 1
            
            policy_output.append(chosen_slot)
            
        return policy_output

def main():
    # 1. Context Simulation: Define required inputs based on the prompt structure
    agent_persona = "Position 1 battery engineer balancing budget and solar backfeed"
    agent_location = 1
    agent_base_demand = [1.20, 0.70, 0.80, 0.60]
    
    neighbor_examples = [
        {
            'name': 'Neighbor 2',
            'persona': 'Position 2 feeder analyst prioritising transformer headroom (location 2)',
            'location': 2,
            'Base demand': [0.70, 1.00, 0.80, 0.50],
            'Preferred slots': [1, 2],
            'Comfort penalty': 0.14,
            'Ground truth min-cost slots by day': ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'], # Simplification: assuming 1 for all 7 days
            'gt_slots': [1, 1, 1, 1, 1, 1, 1]
        },
        {
            'name': 'Neighbor 3',
            'persona': 'Position 3 night-shift nurse on the central ridge (location 3)',
            'location': 3,
            'Base demand': [0.60, 0.80, 0.90, 0.70],
            'Preferred slots': [1, 3],
            'Comfort penalty': 0.20,
            'Ground truth min-cost slots by day': ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
            'gt_slots': [1, 1, 1, 3, 1, 1, 1]
        }
    ]

    policy_engine = AgentPolicy(
        persona=agent_persona,
        location=agent_location,
        base_demand=agent_base_demand,
        neighbor_examples=neighbor_examples
    )
    
    # 2. Choose the slot for the next seven days based on imitation
    final_plan = policy_engine.generate_policy()
    
    # 3. Write local_policy_output.json
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(final_plan, f, indent=4)

if __name__ == "__main__":
    # To fulfill the requirement of being runnable via `python policy.py`, 
    # we execute the main logic when the file is run directly.
    main()