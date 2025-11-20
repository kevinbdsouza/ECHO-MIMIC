import json
import numpy as np

class EVSimPolicy:
    def __init__(self):
        # Load scenario data from scenario.json
        try:
            with open('scenario.json', 'r') as f:
                self.scenario_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("scenario.json not found. Ensure it is in the execution directory.")

        self.agent_id = 5
        self.num_slots = 4
        self.num_days = 7
        
        # Agent 5 specific parameters
        self.base_demand = np.array([0.50, 0.70, 0.60, 0.90])
        self.capacity = self.scenario_data['capacity']
        self.alpha = self.scenario_data['alpha']
        self.beta = self.scenario_data['beta']
        self.gamma = self.scenario_data['gamma']

        # Fixed constraints (from scenario definition)
        self.slot_min_sessions = np.array([
            self.scenario_data['slot_min_sessions'][str(i)] 
            for i in range(self.num_slots)
        ])
        self.slot_max_sessions = np.array([
            self.scenario_data['slot_max_sessions'][str(i)] 
            for i in range(self.num_slots)
        ])
        
        # Parse neighbor examples
        self.neighbors = {}
        for neighbor_data in self.scenario_data['neighbor_examples']:
            name = neighbor_data['name']
            loc = neighbor_data['location']
            self.neighbors[loc] = {
                'base_demand': np.array(neighbor_data['Base demand']),
                'preferred_slots': set(neighbor_data['Preferred slots']),
                'comfort_penalty': neighbor_data['Comfort penalty'],
                'ground_truth_min_cost': neighbor_data['Ground truth min-cost slots by day']
            }

    def _get_day_data(self, day_index):
        """Extracts tariff, carbon, and baseline load for a specific day."""
        day_key = f"Day {day_index + 1}"
        day_info = self.scenario_data['days'][day_key]
        
        # Base data (used as a default/forecast noise estimate)
        base_tariff = np.array(self.scenario_data['price'])
        base_carbon = np.array(self.scenario_data['carbon_intensity'])
        base_baseline = np.array(self.scenario_data['baseline_load'])
        
        # Day-specific data
        day_tariff = np.array(day_info['Tariff'])
        day_carbon = np.array(day_info['Carbon'])
        day_baseline = np.array(day_info['Baseline load'])
        
        # Spatial Carbon (We only care about our location: Agent 5 -> Location 5)
        spatial_carbon_str = day_info['Spatial carbon'].split(';')[self.agent_id - 1]
        spatial_carbon = np.array([float(c) for c in spatial_carbon_str.split(', ')])
        
        # Noise consideration: Use day-specific data as the best forecast,
        # but acknowledge uncertainty by potentially blending or being conservative.
        # For simplicity in this heuristic, we treat the day-specific data as the realized forecast.
        
        return {
            'tariff': day_tariff,
            'carbon': day_carbon,
            'baseline': day_baseline,
            'spatial_carbon': spatial_carbon
        }

    def calculate_agent_cost(self, slot_index, day_data, neighbor_session_counts):
        """
        Calculates the individualized cost function for agent 5 in a given slot.
        Cost = Alpha * Carbon_Metric + Beta * Price + Gamma * Congestion + Comfort_Penalty
        """
        
        # 1. Carbon Cost (Weighted by Agent 5's specific spatial carbon profile)
        # Agent 5 is at Location 5. We use the spatial carbon for Location 5 at this slot.
        spatial_carbon = day_data['spatial_carbon'][slot_index]
        carbon_cost = self.alpha * spatial_carbon
        
        # 2. Price Cost
        price = day_data['tariff'][slot_index]
        price_cost = self.beta * price
        
        # 3. Congestion Cost (Local Transformer Capacity Constraint)
        # This agent's required energy consumption is base_demand[slot_index]
        agent_load = self.base_demand[slot_index]
        
        # Estimate total load for this slot based on neighbors and self
        # Assume neighbors use their baseline + agent_load if they are active.
        # Since we don't know neighbor's future session counts, we must estimate
        # the congestion based on coordination goals.
        
        # In Stage 3 (Collective), session counts are the coordination variable.
        # We assume that if we choose a slot, we contribute 1 session.
        
        # Estimate neighbors' participation based on their preference/ground truth:
        # Neighbor 4 (Location 4) prefers 0, 3. Neighbor 1 (Location 1) prefers 0, 2.
        # Since we don't know the final session allocation, we estimate congestion 
        # based on the sum of sessions *expected* to be scheduled + 1 (for us).
        
        # A simple coordination heuristic: assume neighbors prioritize slots 
        # that look good for them globally, but respect their known preferences.
        
        estimated_total_sessions = 1 # This agent
        
        # For simplicity in Stage 3, we assume neighbors follow their min-cost slot 
        # for the current day if it aligns with their preferences, otherwise we assume 
        # they use the average baseline load distribution for congestion estimation.
        
        # Since we are calculating the cost *if we choose this slot*, we should 
        # look at how much this slot contributes to the *total* expected load, 
        # relative to the capacity.
        
        # Total baseline load for this slot across all 5 locations (5 neighbors + self)
        # NOTE: Neighbor data specifies base demand, not session contribution.
        # We must relate capacity constraint violation to sessions.
        
        # Capacity is 6.8 units. We assume each session consumes some fraction of the remaining baseline capacity.
        
        # Let's use the provided baseline load + neighbor baseline loads to estimate the total expected load if everyone does average charging.
        
        # Total baseline estimate (Self + Neighbors)
        total_baseline_load = day_data['baseline'][slot_index] # Agent 5's baseline contribution
        
        # Add neighbor baseline loads (only 2 neighbors known)
        if 4 in self.neighbors:
            total_baseline_load += self.neighbors[4]['base_demand'][slot_index]
        if 1 in self.neighbors:
            total_baseline_load += self.neighbors[1]['base_demand'][slot_index]
            
        # Estimate *future* sessions based on neighbor examples: 
        # If we assume the coordination results in slots being filled up to max sessions (2),
        # we estimate the load based on average demand contribution per session.
        
        # Since we are making a single choice recommendation, we focus on avoiding 
        # overloading the capacity metric provided in the scenario.
        
        # Congestion metric: How far are we from capacity, assuming this slot is highly utilized?
        # We estimate based on neighbors' historical behavior (ground truth min-cost slots)
        
        # Simplified Congestion: Assume this slot, if chosen, pulls total load towards the capacity limit.
        # We approximate congestion penalty based on how close the collective baseline load + our demand is to capacity.
        
        # We assume capacity is tied to the sum of baseline loads + session loads.
        # Let's simplify: Congestion = max(0, (Total Estimated Load - Capacity))
        
        # Assume 1 session = 0.2 demand unit (arbitrary normalization based on base demand spread)
        # If we choose this slot, we add our base demand contribution.
        estimated_load_if_we_choose = total_baseline_load + self.base_demand[slot_index]
        
        congestion_violation = max(0, estimated_load_if_we_choose - self.capacity)
        congestion_cost = self.gamma * congestion_violation
        
        # 4. Comfort Penalty (Agent 5: Commuting late, values comfort less than budget/carbon, base penalty 0)
        # The prompt implies Agent 5 has no explicitly stated comfort penalty in its persona/profile setup,
        # but since neighbors have them, we assume Agent 5's penalty is implicitly low or zero unless defined.
        # We set comfort penalty to 0 as it's not defined for Agent 5.
        comfort_penalty = 0.0
        
        total_cost = carbon_cost + price_cost + congestion_cost + comfort_penalty
        
        return total_cost

    def calculate_coordination_impact(self, slot_index, day_data, neighbor_session_counts):
        """
        Calculates the positive impact of choosing this slot regarding global goals,
        especially Carbon and Congestion avoidance, relative to the neighborhood.
        
        Coordination Goal: Minimize collective carbon/congestion.
        """
        
        # Global Carbon (Sum of all spatial carbon values, simplified by using the average intensity)
        avg_carbon_intensity = day_data['carbon'][slot_index]
        
        # Global Congestion (If we choose this slot, we increase load by self.base_demand)
        # This is hard to quantify without knowing neighbor session counts for the day.
        # We rely on the local cost function's congestion term to handle capacity, 
        # and use this function primarily for carbon coordination payoff.
        
        # Carbon Payoff: Lower is better.
        # If the slot has low carbon intensity, the payoff is high (negative cost).
        carbon_payoff = -self.alpha * avg_carbon_intensity
        
        # Coordination with Neighbors:
        # Neighbor 4 prefers 0, 3. Neighbor 1 prefers 0, 2.
        # If we choose a slot they avoid, the collective load is spread, which is good.
        
        coordination_bonus = 0.0
        
        # Check if neighbors are likely to use this slot based on their preferences/history
        is_preferred_by_neighbor1 = slot_index in self.neighbors[1]['preferred_slots']
        is_preferred_by_neighbor4 = slot_index in self.neighbors[4]['preferred_slots']
        
        # If the slot is NOT preferred by neighbors, choosing it helps spread the load (Coordination Bonus)
        if not is_preferred_by_neighbor1 and not is_preferred_by_neighbor4:
            coordination_bonus += 0.5 * self.gamma # Moderate bonus for spreading load away from known preferences
            
        # If this slot is historically low-carbon for neighbors (using their ground truth), 
        # we assume it's a good slot generally, so coordination bonus is neutral (we follow the good path).
        
        # Note: Agent 5 is 'Position 5 graduate tenant commuting late', likely prioritizing budget/schedule over comfort, 
        # but the goal is *collective* coordination. We prioritize low global carbon/congestion.
        
        return carbon_payoff + coordination_bonus


    def decide_slot(self):
        
        recommendations = {}
        
        # Determine the recommended session counts for neighbors based on their history/preferences
        # This is a crude estimation of what the collective system might look like.
        
        # 1. Estimate neighbor session counts for the day (Simplification: Assume they pick their Day N GT slot)
        neighbor_session_estimates = {loc: {s: 0 for s in range(self.num_slots)} for loc in self.neighbors}
        
        for day_idx in range(self.num_days):
            day_data = self._get_day_data(day_idx)
            
            # For Agent 5's decision on Day D, we look at the cost/payoff for Day D
            
            best_cost = float('inf')
            best_slot = -1
            
            # 1. Determine the local cost for Agent 5 if it chooses slot S
            local_costs = []
            for s in range(self.num_slots):
                # For local cost calculation, we use the estimated state of the system.
                # Since neighbor_session_counts is not directly provided in this step (it's output for coordination), 
                # we run the local cost calculation assuming a "neutral" environment where only baseline matters, 
                # relying heavily on the environmental metrics (Carbon/Price).
                
                cost = self.calculate_agent_cost(s, day_data, neighbor_session_counts={})
                local_costs.append(cost)
                
            # 2. Determine the coordination payoff for Agent 5 if it chooses slot S
            coordination_payoffs = []
            for s in range(self.num_slots):
                payoff = self.calculate_coordination_impact(s, day_data, neighbor_session_counts={})
                coordination_payoffs.append(payoff)

            # 3. Combined Score: Minimize (Local Cost - Coordination Payoff)
            # We want low cost AND high payoff (low carbon/congestion avoidance)
            combined_scores = [lc - cp for lc, cp in zip(local_costs, coordination_payoffs)]
            
            # 4. Select the best slot based on the combined score, respecting min/max sessions (which are global constraints, not agent constraints here)
            
            min_score = float('inf')
            
            # Simple selection: lowest combined score wins
            for s in range(self.num_slots):
                score = combined_scores[s]
                
                # Apply soft constraints based on Agent 5's general tendency (though collective goal dominates)
                # Agent 5 is a late commuter, so perhaps slots 2/3 (later) are slightly preferred IF costs are equal.
                if score < min_score:
                    min_score = score
                    best_slot = s
                elif score == min_score:
                    # Tie-breaker: Prefer later slots (higher index) if cost is identical
                    if s > best_slot:
                        best_slot = s
                        
            recommendations[f"Day {day_idx + 1}"] = best_slot
            
            # 5. Update estimated session counts for the *next* iteration (though we only need the recommendation for the output)
            # If Agent 5 chose slot `best_slot`, we increment its assumed contribution.
            # For the purpose of this 7-day forecast run, we don't update the session counts 
            # for neighbors as they are also making independent forecasts/decisions based on *their* history.
        
        # Format output
        output_slots = [recommendations[f"Day {i+1}"] for i in range(self.num_days)]
        return output_slots

    def run(self):
        
        # Step 1: Compute 7-day recommendations
        recommended_slots = self.decide_slot()
        
        # Step 2: Create output structure
        output_data = {
            "agent_id": self.agent_id,
            "recommendations": recommended_slots
        }
        
        # Step 3: Save global_policy_output.json
        with open('global_policy_output.json', 'w') as f:
            json.dump(output_data, f, indent=4)
            
        # Step 4: Save policy.py (This script itself)
        # Since this is the final output, we don't need to save the script content here, 
        # as the prompt asks for the script content as the return value.

# --- Execution ---
if __name__ == "__main__":
    policy = EVSimPolicy()
    policy.run()