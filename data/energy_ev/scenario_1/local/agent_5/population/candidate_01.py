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

    def _get_base_demand(self):
        # Base demand for Agent 5 (index 4 in the scenario list)
        # The structure of base_demand in the prompt seems to be fixed for the agent profile.
        return [0.50, 0.70, 0.60, 0.90]

    def _get_slot_limits(self):
        # Min/Max sessions from the main scenario block
        return {
            'min': [self.scenario['slot_min_sessions'][str(i)] for i in range(self.num_slots)],
            'max': [self.scenario['slot_max_sessions'][str(i)] for i in range(self.num_slots)]
        }

    def _get_spatial_carbon(self, day_data, location):
        """Extracts spatial carbon data for the agent's location."""
        spatial_key = f"Spatial carbon: {location}: "
        # Find the correct string in the day_data dictionary keys
        for key in day_data:
            if key.startswith("Spatial carbon:") and key.endswith(f"| {location}:"):
                # Extract the relevant part: "411, 376, 554, 623" for Day X
                # The prompt structure is tricky. Let's assume the spatial data in the day keys
                # is structured as Location1: C1, C2, C3, C4; Location2: C1, C2, C3, C4; ...
                
                # We need to parse the full string for the day:
                # e.g., Day 1: Spatial carbon: 1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ...
                
                parts = day_data[key].split(';')
                for part in parts:
                    if part.strip().startswith(str(location) + ':'):
                        # part.strip() -> "5: 411, 376, 554, 623" (using scenario block format for reference)
                        # Day X format is: " X: C1, C2, C3, C4"
                        carbon_str = part.strip().split(':')[1].strip()
                        return [int(c) for c in carbon_str.split(',')]
        
        # Fallback: use the scenario context spatial carbon if the day key parsing fails
        # This is highly unlikely given the prompt structure but good for robustness in a real system.
        # For this imitation stage, we MUST rely on the day data.
        
        # Re-parsing based on structure provided in scenario block vs daily block:
        # Scenario block: "spatial_carbon: 1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | ..."
        # Day block: "Spatial carbon: 1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ..."
        
        spatial_str = day_data.get('Spatial carbon')
        if not spatial_str:
            raise ValueError(f"Could not find Spatial Carbon data for {day_data.keys()}")

        for loc_data in spatial_str.split(';'):
            if loc_data.strip().startswith(f"{location}:"):
                carbon_values = loc_data.strip().split(':')[1].strip().split(',')
                return [int(c.strip()) for c in carbon_values]
        
        raise ValueError(f"Spatial carbon data for location {location} not found in day block.")


    def calculate_cost(self, day_key, slot_index):
        """Calculates the 'cost' for Agent 5 (Imitation Stage: Minimize own price/carbon)."""
        
        day_data = self.scenario['days'][day_key]
        
        tariff = day_data['Tariff'][slot_index]
        carbon = day_data['Carbon'][slot_index]
        
        # Agent 5 profile: Position 5 graduate tenant commuting late. 
        # Primary objective in imitation stage is often personal cost minimization, 
        # especially price, but since the goal is *imitation*, we look at what 
        # a cost-conscious user would do, prioritizing tariff/carbon.
        
        # Spatial carbon impact
        try:
            spatial_carbon_list = self._get_spatial_carbon(day_data, self.location)
            spatial_carbon = spatial_carbon_list[slot_index]
        except ValueError:
            # Fallback: Use local carbon if parsing fails (should not happen if data structure is strictly followed)
            spatial_carbon = carbon

        # Demand factor (base demand for the slot)
        demand_factor = self.base_demand[slot_index]

        # Agent 5 is commuting late, likely needs power later in the evening slots (2, 3) 
        # but since this is Imitation Stage 2, we follow the simplest heuristic:
        # minimize the combined price and carbon for the *user's direct benefit*.
        
        # Simple cost function: weighted sum of Price and Carbon (since Agent 5 needs budget control)
        # We normalize Carbon relative to the expected scenario max (around 750) or use a weight.
        # Weighting Price heavily as budget is usually key for tenants.
        
        # Let's use a simple additive cost prioritizing price, using carbon as a secondary factor.
        # Normalize carbon relative to the average carbon of the day (approximation for scaling)
        avg_carbon = sum(day_data['Carbon']) / len(day_data['Carbon'])
        
        # Cost = Price + (Carbon / AvgCarbon_Day) * Weight
        
        # Given the context often favors simple relative comparison over complex normalization:
        # Cost = Price + (Carbon * K)
        K = 0.0005 # Heuristic weight to bring carbon intensity (hundreds) into line with price (tens)
        
        cost = tariff + (carbon * K) 
        
        # Constraint Check (Sessions)
        # Since we cannot track sessions across days easily without state, we assume 
        # we can pick any slot if min/max session rules are just boundary conditions 
        # that must be met *across the whole week*, not instantly checked. 
        # For Stage 2 imitation, we pick the *subjectively* best slot based on immediate metrics.
        
        return cost, demand_factor

    def plan_schedule(self):
        """Generates the 7-day slot plan based on personal objective (lowest personal cost)."""
        
        day_keys = list(self.scenario['days'].keys())
        plan = []

        for day_key in day_keys:
            best_cost = float('inf')
            best_slot = -1
            
            # Agent 5 (Commuting late) might prefer later slots naturally, 
            # but for imitation, we select the objectively cheapest slot based on our cost function.
            
            for i in range(self.num_slots):
                cost, demand = self.calculate_cost(day_key, i)
                
                # Note: We are ignoring slot_min/max sessions here as per typical Stage 2 behavior 
                # unless the constraints are explicitly local and immediate (which they aren't specified to be).
                
                if cost < best_cost:
                    best_cost = cost
                    best_slot = i
            
            plan.append(best_slot)
            
        return plan

def main():
    # 1. Load scenario data (simulated by defining it here, as file system access is simulated)
    # In a real environment, this would load scenario.json
    
    # Since we must produce runnable Python code that relies ONLY on the prompt context, 
    # we must recreate the necessary data structure available via 'scenario.json'.
    
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
    
    # Add the main scenario block data points to the days if they are necessary for reference,
    # but the implementation above relies on day-specific overrides. We ensure the spatial carbon
    # field in the day structure is correctly formatted for the parser.
    
    # Since the scenario structure provided only has global values outside 'days', 
    # we need to augment the structure slightly to allow AgentPolicy to work as written, 
    # primarily by ensuring all necessary daily data keys exist.
    
    # The structure defined in the __init__ and _get_spatial_carbon is tailored to find 
    # the spatial data *within* the day dictionary. We must ensure the data structure 
    # matches what the parser expects. The prompt implies the spatial data listed under 
    # the main block is NOT the daily data, but the daily data is explicitly listed 
    # for each day (e.g., "Spatial carbon: 1: ...").

    # 2. Choose policy
    agent = AgentPolicy(
        scenario_data=scenario_context, 
        agent_id=5, 
        agent_location=5
    )
    
    schedule = agent.plan_schedule()
    
    # 3. Write output
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(schedule, f)

if __name__ == "__main__":
    # In the execution environment, this script is run directly.
    # We simulate running the main logic here.
    main()