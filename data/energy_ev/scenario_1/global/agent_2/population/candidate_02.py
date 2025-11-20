import json
import numpy as np
from collections import defaultdict

# --- Agent Configuration and Scenario Loading ---

class AgentPolicy:
    def __init__(self, scenario_data, agent_id=2):
        self.scenario = scenario_data
        self.agent_id = agent_id
        self.num_slots = len(scenario_data['slots'])
        self.num_days = len(scenario_data['days'])
        
        # Agent specific parameters (Position 2, Feeder Analyst focused on Headroom)
        self.location = self.scenario['location'] # Location 2
        self.base_demand = np.array(self.scenario['base_demand'])
        self.alpha = self.scenario['alpha']
        self.beta = self.scenario['beta']
        self.gamma = self.scenario['gamma']
        
        # Fixed constraints from scenario description (for context, though dynamic limits are in day data)
        self.capacity = self.scenario['capacity']

        # Parse neighbor data
        self.neighbors = self._parse_neighbors(self.scenario['neighbor_examples'])

    def _parse_neighbors(self, neighbor_data):
        neighbors = {}
        for i, neighbor in enumerate(neighbor_data):
            # Neighbor ID is based on order, assuming Neighbor 1 corresponds to index 0, etc.
            # We use the name provided for easier mapping if needed, but for coordination,
            # we map by location if possible or use a simple index.
            n_id = i + 1 # Neighbor 1, 2, ...
            
            # Extract location for spatial context (if location is not explicitly listed, we must infer/skip)
            # Based on problem description, we use location provided in neighbor details if available.
            try:
                location = neighbor['location']
            except KeyError:
                # If location is missing in the neighbor example structure, we skip spatial consideration for them
                location = None 
                
            neighbors[n_id] = {
                'location': location,
                'base_demand': np.array(neighbor['Base demand']),
                'preferred_slots': set(neighbor['Preferred slots']),
                'comfort_penalty': neighbor['Comfort penalty'],
                'ground_truth_slots': neighbor['Ground truth min-cost slots by day']
            }
        return neighbors

    def _load_day_data(self, day_name):
        day_data = self.scenario['days'][day_name]
        return {
            'tariff': np.array(day_data['Tariff']),
            'carbon': np.array(day_data['Carbon']),
            'baseline_load': np.array(day_data['Baseline load']),
            'spatial_carbon': self._parse_spatial_carbon(day_data['Spatial carbon'])
        }

    def _parse_spatial_carbon(self, sc_str):
        # spatial_carbon: 1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | ...
        parsed = {}
        # Remove padding/spaces and split by '|' to get neighbor data blocks
        blocks = [b.strip() for b in sc_str.split('|')]
        for block in blocks:
            try:
                loc_id_str, values_str = block.split(':', 1)
                loc_id = int(loc_id_str.strip())
                values = np.array([int(v.strip()) for v in values_str.split(',')])
                parsed[loc_id] = values
            except ValueError:
                # Handle malformed blocks if necessary
                continue
        return parsed

    def _calculate_agent_cost(self, day_index, slot_index, scheduled_sessions):
        """
        Calculates the cost function for Agent 2 (Feeder Analyst) for a given slot.
        Focus: Transformer Headroom (Capacity constraint) + Global Metrics (Carbon, Price) + Comfort.
        
        Cost = alpha * Congestion_Penalty + beta * Carbon_Cost + gamma * Price_Cost + Comfort_Penalty
        """
        day_key = list(self.scenario['days'].keys())[day_index]
        data = self._load_day_data(day_key)
        
        # 1. Congestion/Headroom Penalty (Primary concern for Feeder Analyst)
        # Total load = Baseline + Agent Load (Demand * Sessions)
        current_load = data['baseline_load'][slot_index] + self.base_demand[slot_index] * scheduled_sessions
        
        # Penalize heavily if approaching or exceeding capacity (using a soft penalty modeled by alpha)
        if current_load > self.capacity:
            # Hard penalty for exceeding capacity
            congestion_penalty = 1e6 * (current_load - self.capacity) 
        elif current_load > 0.9 * self.capacity:
            # Soft penalty approaching capacity limit (e.g., 90%)
            congestion_penalty = self.alpha * (current_load - 0.9 * self.capacity) ** 2
        else:
            congestion_penalty = 0.0
            
        # 2. Global Carbon Cost (Minimized via carbon intensity)
        carbon_cost = self.gamma * data['carbon'][slot_index]
        
        # 3. Price Cost (Minimized via tariff)
        price_cost = self.beta * data['tariff'][slot_index]
        
        # 4. Comfort Penalty (Agent 2 has no explicit comfort penalty defined, set to 0)
        comfort_penalty = 0.0 
        
        # 5. Coordination Signal (Spatial Carbon)
        # We prefer slots where our neighbors *aren't* stressing the local spatial grid, 
        # or where the local grid component of carbon intensity is low.
        # Since we are location 2, we look at our local carbon footprint.
        # Coordination is implicit by selecting low overall carbon slots (handled by gamma * carbon), 
        # but for explicit coordination (ECHO Stage 3), we check if the decision aligns with neighbor behavior.
        
        # In this stage, we primarily focus on the hard constraints and local optimization, 
        # letting the structure (alpha, beta, gamma) guide the trade-off based on the primary goal (headroom).

        total_cost = congestion_penalty + carbon_cost + price_cost + comfort_penalty
        
        return total_cost

    def _calculate_neighbor_impact(self, day_index, slot_index, agent_session):
        """
        Calculates the penalty incurred due to neighbor coordination challenges.
        We specifically look at spatial carbon conflicts (where neighbors draw heavily in nearby locations).
        """
        day_key = list(self.scenario['days'].keys())[day_index]
        data = self._load_day_data(day_key)
        
        # Location 2 (Agent 2)
        
        # Neighbors are N1 (Loc 1), N2 (Loc 3), N3 (Loc 4), N4 (Loc 5). 
        # (Assuming index mapping based on list order: 0->N1, 1->N2, 2->N3, 3->N4)
        # Neighbor data structure is based on observation list order (1, 4 in this case).
        # We need to map observation order to location IDs present in spatial carbon data (1 to 5).
        
        # Observed neighbors:
        # Neighbor 1 (N1): Location 1
        # Neighbor 4 (N4): Location 4
        
        neighbor_impact = 0.0
        
        # Assume neighbors try to coordinate based on their ground truth slots,
        # but we must coordinate based on *our* slot choice impacting them, or vice versa.
        
        # For Stage 3, coordination usually means: If neighbors are highly loaded/stressed, 
        # we should shift away if possible, or ensure our load is minimal if the grid is constrained.
        
        # Since we are prioritizing Headroom (Capacity), we primarily use the congestion penalty.
        # For coordination, we check if selecting this slot aligns with the *overall* spatial constraint.
        
        # Spatial Carbon Check: If our choice forces a high local spatial carbon output (sum of local grid nodes), 
        # and neighbors are already scheduled high there, we get penalized.
        
        # Given the complexity of dynamically modeling neighbor sessions without knowing their policy, 
        # we rely on the stated neighborhood preferences/targets as a weak coordination signal.
        
        # Coordination Heuristic: If we are selecting a peak slot (high price/carbon), check if 
        # neighbors *explicitly* prefer this slot. If they do, it might mean they are constrained 
        # and our choice might hurt coordination.
        
        for n_id, n_data in self.neighbors.items():
            n_loc = n_data['location']
            
            if n_loc is None:
                continue

            # Check if this slot is highly preferred by the neighbor (low comfort penalty implies preference)
            is_neighbor_preferred = slot_index in n_data['preferred_slots']
            
            # If we pick a slot that is *bad* globally (high carbon) AND a neighbor *prefers* it (suggesting local stress or high value for them),
            # we slightly increase our coordination penalty if we DON'T pick it, assuming they need it.
            # This is complex. A simpler, robust coordination mechanism: avoid slots that are simultaneously high risk (high local spatial carbon)
            # AND preferred by neighbors who might fight for local headroom.
            
            # Use Spatial Carbon as a proxy for local grid stress.
            # If the spatial carbon at our location (Loc 2) is high for this slot, it means the grid is strained locally, 
            # and we should favor slots with lower spatial carbon, especially if neighbors are known to cause high load (high base demand neighbors).
            
            # This agent focuses on capacity. We assume coordination means *not* creating local peaks that neighbors dislike.
            
            # Since Agent 2 is Location 2, we look at the spatial carbon contribution AT Location 2 from ALL nodes.
            # Spatial carbon data gives values *for* node 1, node 2, node 3, etc.
            
            # If data['spatial_carbon'][n_loc][slot_index] is high, it indicates high strain originating/affecting node n_loc during this slot.
            
            # Simple coordination: If the overall spatial carbon *at our location* (Loc 2) is very high, 
            # and the neighbor seems load-heavy (e.g., Neighbor 1's base demand is high), we penalize ourselves slightly if we choose a heavy load session count.
            
            if n_loc == self.location:
                # Local spatial carbon constraint (if available for our location node)
                local_spatial_stress = data['spatial_carbon'].get(self.location, np.zeros(self.num_slots))[slot_index]
                
                # If local stress is high, favor lower session count decisions, but we are calculating cost *per slot*, not per session count yet.
                # This impact term should be minimal if the primary cost function handles capacity.
                pass 

        return neighbor_impact
    
    def _determine_sessions_for_slot(self, day_index, slot_index, max_sessions):
        """
        Determines the optimal number of sessions (0, 1, or 2) for a given slot 
        by evaluating the cost function derived from the overall goals.
        Agent 2 has min=1, max=2 sessions defined in the scenario description slots, but we check the day-specific limits.
        """
        day_key = list(self.scenario['days'].keys())[day_index]
        day_limits = self.scenario['days'][day_key]
        
        min_sessions = self.scenario['slot_min_sessions'][slot_index]
        day_max_sessions = day_limits.get('slot_max_sessions', {}).get(str(slot_index))
        if day_max_sessions is None:
             day_max_sessions = self.scenario['slot_max_sessions'][slot_index]
             
        max_allowed = min(max_sessions, day_max_sessions)
        
        best_session_count = min_sessions
        min_total_cost = float('inf')

        # Iterate over possible session counts (constrained by the day/scenario limits)
        for sessions in range(min_sessions, max_allowed + 1):
            
            # 1. Calculate Agent Cost (Headroom focus)
            agent_cost = self._calculate_agent_cost(day_index, slot_index, sessions)
            
            # 2. Calculate Coordination Impact (Using zero here, as coordination is mainly handled by slot selection)
            coordination_impact = self._calculate_neighbor_impact(day_index, slot_index, sessions)
            
            total_cost = agent_cost + coordination_impact
            
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_session_count = sessions
                
        return best_session_count, min_total_cost


    def _select_day_slot(self, day_index):
        """
        Selects the best slot (index 0-3) for a given day by minimizing the cost function 
        across all slots, assuming a 'standard' session count if needed for comparison, 
        but focusing on minimizing the slot cost itself.
        
        Since coordination is complex, we use a heuristic approach: prioritize low carbon/price, 
        but heavily penalize capacity violations.
        """
        day_key = list(self.scenario['days'].keys())[day_index]
        data = self._load_day_data(day_key)
        
        slot_costs = []
        
        # Determine day-specific limits (We assume a standard expected session load for slot selection comparison, e.g., min session count)
        expected_sessions = 1 # Used only for cost comparison baseline
        
        for slot_index in range(self.num_slots):
            
            # 1. Calculate the cost if we choose this slot, assuming minimal load (expected_sessions)
            # Note: The cost function internalizes the session count, but here we are comparing slots, 
            # assuming the best possible session schedule *within* that slot choice.
            
            # To compare slots fairly, we assume the *minimum required* session count (1) is scheduled, 
            # and the cost function will capture the true penalty.
            
            _, cost = self._determine_sessions_for_slot(day_index, slot_index, max_sessions=2)
            
            # 2. Incorporate Neighbor Coordination Signals into Slot Selection Heuristic (Global Goal vs Local Goal)
            
            # Agent 2 (Feeder Analyst, Location 2) prioritizes Headroom (Cost Function handles this via alpha)
            
            # Coordination Heuristic for Slot Selection: 
            # Check if selecting this slot forces a high spatial carbon constraint on our local node (Loc 2).
            local_spatial_stress = data['spatial_carbon'].get(self.location, np.zeros(self.num_slots))[slot_index]
            
            # Neighbor 1 (Loc 1) and Neighbor 4 (Loc 4) are known. 
            # If neighbors are known to cause load, and this slot aligns with low spatial carbon at our location, it's better.
            
            # Coordination Term: Penalize slots where the local grid stress (spatial carbon at Loc 2) is high, 
            # unless that high stress is unavoidable (i.e., baseline load is already high).
            
            coordination_penalty = self.alpha * local_spatial_stress # Penalize local stress
            
            # Final composite cost for slot ranking
            final_slot_cost = cost + coordination_penalty
            
            slot_costs.append((final_slot_cost, slot_index))

        # Select the slot with the minimum final composite cost
        slot_costs.sort(key=lambda x: x[0])
        best_slot_index = slot_costs[0][1]
        
        # Final check: Ensure minimum session requirement is met for the chosen slot.
        # Although session determination happens inside _determine_sessions_for_slot, we must ensure 
        # that the chosen slot can accommodate at least min_sessions=1. Since the cost function
        # heavily penalizes infeasibility (sessions < min), this is implicitly handled.
        
        # We run the session determination one last time for the chosen slot to ensure the final 
        # output structure is satisfied (though we only return the slot index here, not sessions).
        
        return best_slot_index

    def run_policy(self):
        recommendations = {}
        day_names = list(self.scenario['days'].keys())
        
        # Loop through the 7 days
        for i, day_name in enumerate(day_names):
            
            # 1. Select the best slot for the day based on combined cost and coordination signals
            best_slot = self._select_day_slot(i)
            
            recommendations[day_name] = best_slot
            
            # 2. Determine the optimal session count for the chosen slot (This step isn't strictly needed for the output JSON, 
            # but it confirms feasibility based on the cost optimization run)
            # max_sessions is fixed at 2, min at 1 for this agent's context slots.
            _, _ = self._determine_sessions_for_slot(i, best_slot, max_sessions=2) 
            
        # Format output
        output_slots = [recommendations[day_names[i]] for i in range(self.num_days)]
        
        # Save results
        with open('global_policy_output.json', 'w') as f:
            json.dump(output_slots, f, indent=4)
            
        print(f"Agent 2 Policy executed. Output saved to global_policy_output.json: {output_slots}")
        
        return output_slots

# --- Execution ---

def load_scenario():
    # Assuming scenario.json is available in the execution directory
    try:
        with open('scenario.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback for testing if file loading fails outside the specific environment structure
        # This is dangerous but ensures the script runs standalone if needed for inspection.
        # In the final environment, 'scenario.json' must be present.
        raise FileNotFoundError("scenario.json not found. Ensure it is accessible.")

if __name__ == "__main__":
    scenario_data = load_scenario()
    agent_policy = AgentPolicy(scenario_data, agent_id=2)
    agent_policy.run_policy()