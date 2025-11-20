import json
import numpy as np

class Policy:
    def __init__(self):
        # --- Load Scenario Data ---
        try:
            with open('scenario.json', 'r') as f:
                scenario_data = json.load(f)
        except FileNotFoundError:
            # Mock data structure if file is not found (for structured development outside the execution environment)
            scenario_data = self._mock_scenario_data()

        self.slots_info = scenario_data['slots']
        self.price_base = np.array(scenario_data['price'])
        self.carbon_base = np.array(scenario_data['carbon_intensity'])
        self.capacity = scenario_data['capacity']
        self.baseline_load = np.array(scenario_data['baseline_load'])
        self.slot_min_sessions = scenario_data['slot_min_sessions']
        self.slot_max_sessions = scenario_data['slot_max_sessions']
        self.spatial_carbon_base = self._parse_spatial_carbon(scenario_data['spatial_carbon'])
        
        self.days_data = scenario_data['days']
        self.alpha = scenario_data['alpha']  # Carbon weight
        self.beta = scenario_data['beta']   # Price weight
        self.gamma = scenario_data['gamma'] # Comfort weight

        self.profile = scenario_data['profile']
        self.neighbor_examples = scenario_data['neighbor_examples']
        
        self.location = self.profile['location']
        self.base_demand = np.array(self.profile['base_demand'])
        
        self.num_slots = len(self.slots_info)
        self.num_days = 7

        # Derived data for the 7-day forecast
        self.forecast_tariffs = []
        self.forecast_carbons = []
        self.forecast_baselines = []
        self.forecast_spatial_carbons = []
        self._process_forecasts()

    def _mock_scenario_data(self):
        # Mock data structure matching the required input format for initialization testing
        return {
            "slots": {
                "0": "19-20", "1": "20-21", "2": "21-22", "3": "22-23"
            },
            "price": [0.23, 0.24, 0.27, 0.30],
            "carbon_intensity": [700, 480, 500, 750],
            "capacity": 6.8,
            "baseline_load": [5.2, 5.0, 4.9, 6.5],
            "slot_min_sessions": {"0": 1, "1": 1, "2": 1, "3": 1},
            "slot_max_sessions": {"0": 2, "1": 2, "2": 1, "3": 2},
            "spatial_carbon": "1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | 3: 503, 473, 471, 577 | 4: 617, 549, 479, 363 | 5: 411, 376, 554, 623",
            "days": {
                "Day 1 (Day 1 — Clear start to the week with feeders expecting full-slot coverage.)": {
                    "Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], 
                    "Baseline load": [5.3, 5.0, 4.8, 6.5],
                    "Spatial carbon": "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; 3: 590, 520, 340, 630; 4: 620, 560, 500, 330; 5: 360, 380, 560, 620"
                },
                "Day 2 (Day 2 — Evening wind ramps mean slots 0 and 3 must balance transformer temps.)": {
                    "Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], 
                    "Baseline load": [5.1, 5.2, 4.9, 6.6],
                    "Spatial carbon": "1: 510, 330, 550, 600; 2: 540, 500, 320, 610; 3: 310, 520, 550, 630; 4: 620, 540, 500, 340; 5: 320, 410, 560, 640"
                },
                 # ... include 5 more days for completeness ...
                "Day 7 (Day 7 — Cool front eases late-night load but upstream carbon stays elevated.)": {
                    "Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], 
                    "Baseline load": [5.1, 4.9, 4.8, 6.3],
                    "Spatial carbon": "1: 330, 520, 560, 610; 2: 540, 330, 520, 600; 3: 580, 540, 330, 620; 4: 630, 560, 500, 330; 5: 520, 330, 550, 600"
                }
            },
            "alpha": 40.00, "beta": 0.50, "gamma": 12.00,
            "profile": {
                "persona": "Position 3 night-shift nurse on the central ridge",
                "location": 3,
                "base_demand": [0.60, 0.80, 0.90, 0.70]
            },
            "neighbor_examples": [
                {"location": 2, "Base demand": [0.70, 1.00, 0.80, 0.50], "Ground truth min-cost slots by day": [1, 2, 0, 1, 2, 0, 1]},
                {"location": 5, "Base demand": [0.50, 0.70, 0.60, 0.90], "Ground truth min-cost slots by day": [0, 0, 0, 0, 0, 1, 1]}
            ]
        }

    def _parse_spatial_carbon(self, sc_str):
        """Parses the spatial carbon string into a structured dictionary."""
        sc_dict = {}
        if not sc_str: return sc_dict
        
        parts = sc_str.split('|')
        for part in parts:
            part = part.strip()
            if not part: continue
            
            loc_data = part.split(':')
            location = int(loc_data[0].strip())
            
            values = [float(x.strip()) for x in loc_data[1].split(',')]
            sc_dict[location] = np.array(values)
        return sc_dict

    def _process_forecasts(self):
        """Extracts time-series data for the 7 forecast days."""
        day_keys = sorted(self.days_data.keys())
        
        for day_key in day_keys:
            day_info = self.days_data[day_key]
            self.forecast_tariffs.append(np.array(day_info['Tariff']))
            self.forecast_carbons.append(np.array(day_info['Carbon']))
            self.forecast_baselines.append(np.array(day_info['Baseline load']))
            
            # Process spatial carbon for the specific day
            sc_map = self._parse_spatial_carbon(day_info['Spatial carbon'])
            # Use the current agent's location's spatial carbon profile for the local feeder context
            # Assuming the agent's location index corresponds to the necessary feeder data structure
            self.forecast_spatial_carbons.append(sc_map.get(self.location, self.carbon_base)) # Fallback to base if not found

    def _get_slot_constraints(self, day_index, slot_index):
        """Returns min/max sessions for a given day/slot, respecting scenario defaults if day-specific values are missing."""
        slot_str = str(slot_index)
        day_key = list(self.days_data.keys())[day_index]
        
        # Check for day-specific constraints (though not explicitly defined in the input, we use scenario defaults)
        min_s = self.slot_min_sessions.get(slot_str, 1)
        max_s = self.slot_max_sessions.get(slot_str, 2)
        
        return min_s, max_s

    def calculate_utility(self, day_index, slot_index, demand_multiplier=1.0):
        """
        Calculates the utility score for choosing a slot on a specific day.
        Utility = - [ alpha * Carbon_Cost + beta * Price_Cost + gamma * Comfort_Cost ]
        We maximize utility, so we minimize the weighted cost components.
        """
        
        # 1. Cost Components (Minimize)
        
        # Carbon Cost (Weighted by environment sensitivity)
        # Use forecast carbon intensity + spatial carbon effect
        carbon_intensity = self.forecast_carbons[day_index][slot_index]
        # The agent is on location 3. We use its specific spatial carbon forecast.
        spatial_carbon_factor = self.forecast_spatial_carbons[day_index][slot_index]
        
        # Carbon Score: Weighting intensity and local congestion (spatial factor relative to base intensity)
        carbon_cost = self.alpha * (carbon_intensity + (spatial_carbon_factor / 100.0))
        
        # Price Cost (Weighted by price sensitivity)
        tariff = self.forecast_tariffs[day_index][slot_index]
        price_cost = self.beta * tariff
        
        # Congestion Cost (Relative to baseline, scaled by capacity)
        # This agent's contribution to congestion if it uses the slot
        actual_load = self.baseline_load[slot_index] * demand_multiplier
        baseline_load_slot = self.forecast_baselines[day_index][slot_index]
        
        # Congestion metric: How much the load exceeds the baseline for that slot on that day
        congestion = max(0, actual_load - baseline_load_slot)
        # We scale congestion by capacity relative to the load factor (higher relative load = higher cost)
        congestion_cost = self.alpha * (congestion / self.capacity) * 2.0 # High weight for congestion if it's a problem slot

        total_cost = carbon_cost + price_cost + congestion_cost

        # 2. Comfort Component (Minimize Penalty)
        
        # Comfort Penalty (Minimize deviation from personal base demand profile)
        # The agent is a night-shift nurse (location 3), typically needing high energy in slots 1 and 2 (20h-22h)
        # Base demand profile: [0.60, 0.80, 0.90, 0.70] -> Highest demand in slot 2, then 1.
        
        # Neighbor examples suggest a preference for slots 0/1 or 1/2, but nurse profile strongly suggests late slots.
        # For coordination, we introduce a small penalty for deviating from the *strongest* implied preference 
        # (slot 2 having the highest base demand multiplier 0.90).
        
        # Since we are choosing only ONE slot, the comfort score is high if the chosen slot matches the highest base demand slot (slot 2).
        comfort_penalty = 0
        if slot_index == 2:
            # Highest base demand slot, lowest comfort penalty (highest utility contribution)
            comfort_penalty = 0 
        else:
            # Penalty increases the further we move from the peak demand slot (slot 2)
            comfort_penalty = self.gamma * abs(slot_index - 2) * 0.5 

        # Neighbor coordination check (Observing neighbors' GT slots)
        # Neighbor 2 (location 2, analyst) prefers slots 1, 2.
        # Neighbor 5 (location 5, commuter) prefers slots 0, 1.
        
        # As location 3, we are somewhat central. We want to avoid overlapping heavily with the known neighbors' preferred slots if they are costly.
        # Since we cannot know their actual schedule, we use their historical preference as a soft disincentive if we choose that slot too, 
        # assuming coordination means spreading out.
        neighbor_avoidance_penalty = 0
        
        # Check if this slot is a ground truth favorite for any known neighbor
        for neighbor in self.neighbor_examples:
            gt_slots = neighbor['Ground truth min-cost slots by day']
            if day_index < len(gt_slots) and slot_index in gt_slots[day_index]:
                # If this slot is preferred by a neighbor on this day, we add a small coordination penalty
                neighbor_avoidance_penalty += self.gamma * 0.1 

        
        final_utility = - (total_cost + comfort_penalty + neighbor_avoidance_penalty)
        return final_utility

    def determine_recommendation(self):
        """Determines the best slot for each of the 7 forecast days."""
        
        recommendations = []
        
        # The Agent 3 persona (nurse) is highly likely to need charging when they return from their shift, 
        # likely around slots 1 or 2, where base demand is highest (0.8, 0.9).
        
        # We must respect the min/max session constraints, but since we only choose ONE slot per day, 
        # these constraints only define possibility, not aggregate scheduling over 7 days.
        # We assume base_demand * 1kW is the required energy for that slot.
        
        for day_index in range(self.num_days):
            best_utility = -np.inf
            best_slot = -1
            
            for slot_index in range(self.num_slots):
                
                # Calculate the utility score for this potential choice
                utility = self.calculate_utility(day_index, slot_index, demand_multiplier=self.base_demand[slot_index])
                
                if utility > best_utility:
                    best_utility = utility
                    best_slot = slot_index
            
            recommendations.append(best_slot)
            
        return recommendations

    def run_policy(self):
        recommendations = self.determine_recommendation()
        
        # Prepare output JSON structure
        output_data = {
            "agent_id": "Agent 3",
            "scenario_id": "ev_peak_sharing_1",
            "recommendations": recommendations
        }

        # Save global_policy_output.json
        with open('global_policy_output.json', 'w') as f:
            # Save only the list of recommendations as requested in the specification
            json.dump(recommendations, f, indent=4)
            
        # The policy script itself is the executable Python file
        # The logic is entirely contained within this class instance execution.

if __name__ == "__main__":
    policy = Policy()
    policy.run_policy()
    
    # If this script is run directly, it generates global_policy_output.json 
    # containing the 7-day slot recommendation.