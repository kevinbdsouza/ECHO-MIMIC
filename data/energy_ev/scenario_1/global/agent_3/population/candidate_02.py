import json
import os
from typing import List, Dict, Any

class Policy:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.data = self._load_scenario_data()
        self.params = self.data['alpha_beta_gamma']
        self.scenario_params = self.data['scenario_parameters']
        self.days_data = self.data['days']
        self.neighbor_examples = self.data.get('neighbor_examples', [])
        
        # Agent-specific data derived from persona/profile
        self.base_demand = self.data['profile']['base_demand']
        self.comfort_penalty_base = self.params['beta'] # Using beta as a general comfort factor proxy
        self.location = self.data['profile']['location']

        # Pre-process neighbor data for easier access
        self.neighbor_data = self._process_neighbor_examples()

    def _load_scenario_data(self) -> Dict[str, Any]:
        # Determine the path based on the agent ID structure (Agent 3 in Stage 3)
        # Assuming the structure is consistent for loading scenario.json relative to policy.py execution
        try:
            # Load scenario.json which must be present in the execution directory
            with open('scenario.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: scenario.json not found. Ensure it is in the execution directory.")
            # In a real environment, this would need a more robust path resolution
            # For this exercise, we rely on the file being present.
            raise

    def _process_neighbor_examples(self) -> Dict[int, Dict]:
        processed = {}
        for i, neighbor in enumerate(self.neighbor_examples):
            # Assuming neighbor ID is implicit (1, 2, ...) based on order, or should be extracted if present
            # Since agent IDs aren't explicitly listed for neighbors, we use an index starting from 1, 
            # and rely on 'location' for spatial context.
            neighbor_id = i + 1 
            processed[neighbor_id] = {
                'location': neighbor['location'],
                'base_demand': neighbor['Base demand'],
                'preferred_slots': neighbor['Preferred slots'],
                'comfort_penalty': neighbor['Comfort penalty'],
                'ground_truth_slots': neighbor['Ground truth min-cost slots by day']
            }
        return processed

    def _calculate_slot_utility(self, day_key: str, slot_idx: int, current_day_data: Dict[str, Any]) -> float:
        """
        Calculates a utility score for a given slot on a specific day.
        Utility is maximized when the score is minimized (Lower score = Better).
        Score components:
        1. Comfort/Demand Penalty (Local)
        2. Price Penalty (Local/Global)
        3. Carbon Intensity Penalty (Global)
        4. Congestion/Spatial Penalty (Local/Coordination)
        """
        
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        capacity = self.scenario_params['capacity']
        
        tariffs = current_day_data['Tariff']
        carbons = current_day_data['Carbon']
        baselines = current_day_data['Baseline load']
        
        # Get spatial carbon data for this agent's location (self.location)
        spatial_carbon_str = current_day_data['Spatial carbon'][str(self.location)]
        spatial_carbons = [float(x) for x in spatial_carbon_str.split('; ')]

        # --- 1. Local Comfort / Demand Penalty ---
        # For this persona (Night-shift nurse, high base demand in late slots 2, 3), 
        # slots 0/1 are likely less comfortable/convenient if work ends late, but let's check base demand.
        # Base demand: 0.60 (S0), 0.80 (S1), 0.90 (S2), 0.70 (S3) -> Highest demand in Slot 2.
        
        agent_demand = self.base_demand[slot_idx]
        
        # Comfort is often inverse to demand if demand reflects required activity time.
        # Since the agent is a night-shift nurse, perhaps later slots (S2, S3) are less flexible due to sleep schedule.
        # Let's assume penalty is higher when demand is high OR when it deviates from expected behavior (which we can't fully model here).
        
        # A simple inverse relationship to base demand might represent required activity timing inconvenience.
        # Penalty increases if demand is high (meaning the agent *needs* the slot).
        comfort_penalty = beta * agent_demand
        
        # --- 2. Price Penalty (Cost) ---
        price = tariffs[slot_idx]
        price_penalty = alpha * price

        # --- 3. Carbon Intensity Penalty (Global Goal) ---
        carbon_intensity = carbons[slot_idx]
        carbon_penalty = gamma * carbon_intensity

        # --- 4. Congestion / Spatial Penalty (Coordination/Local Constraint) ---
        # Agent location is 3. We look at the spatial carbon at location 3 for this slot.
        # High spatial carbon indicates localized congestion/stress.
        spatial_stress = spatial_carbons[slot_idx]
        
        # Coordination Heuristic: Coordinate based on neighbors' revealed ground truth *if* they are stressed.
        # This agent (Loc 3) should try to cooperate with known neighbors (Loc 2, Loc 5) to avoid simultaneous high load, 
        # especially if their preferred slots conflict with this agent's preferred low-carbon slots.
        
        # Since the agent's primary goal is low carbon/price, we use spatial carbon as a proxy for local congestion.
        congestion_penalty = 0.1 * spatial_stress # Lower weight than global goals

        # --- Combined Score (Minimize this score) ---
        total_score = (
            price_penalty + 
            carbon_penalty + 
            comfort_penalty + 
            congestion_penalty
        )
        
        # Optional Coordination Adjustment: Penalize slots heavily used by neighbors who have historically favored them
        # Here, we check if this slot conflicts with neighbors' known preferred slots or ground truth.
        neighbor_conflict_penalty = 0.0
        for n_id, n_data in self.neighbor_data.items():
            # We primarily care about coordination regarding the *Global Goals* (Carbon/Price)
            # If a neighbor's ground truth slot is *this* slot, we don't penalize, as that represents known behavior.
            # However, if the neighbor has a *different* preferred slot, and this slot is globally bad (high carbon), 
            # we might want to avoid it if the neighbor *isn't* using it, encouraging them to use their preference instead.
            
            # For simplicity in Stage 3 Collective, we'll primarily focus on spreading load *if* the slot is globally bad.
            # Check if neighbor 2 (Feeder Analyst, Loc 2) prefers slot 1/2.
            # Check if neighbor 5 (Graduate, Loc 5) prefers slot 0/1.
            
            # Since we don't know what neighbors *will* do, we only use Ground Truth slots to estimate expected collective behavior for coordination.
            # If a neighbor used a globally good slot yesterday, we might try to match that pattern *if* it aligns with our comfort.
            
            # Coordination attempt: If this slot is globally high carbon, and a neighbor has an alternative low-carbon GT slot, 
            # we might slightly favor this slot if the neighbor *isn't* using it, assuming they are coordinating elsewhere.
            
            # Given the goal: "satisfy common global goals (carbon, congestion) and not just personal ones, coordinating only with the neighbour information you can observe."
            # We will try to avoid the slot if it is *globally worse* than the best available slot, UNLESS a neighbor seems to *require* it (high GT usage).
            
            pass # Keeping coordination simple and implicit via shared penalties (Carbon/Congestion) unless strong conflict is obvious.

        return total_score

    def _determine_daily_slot(self, day_name: str, day_data: Dict[str, Any]) -> int:
        """
        Selects the single best slot for the given day based on calculated utility scores.
        """
        num_slots = len(self.scenario_params['price'])
        scores = []

        for i in range(num_slots):
            score = self._calculate_slot_utility(day_name, i, day_data)
            
            # Apply constraints: Slot Min/Max sessions (although this agent is single-session, constraints guide optimization)
            slot_min_sessions = self.scenario_params['slot_min_sessions'][str(i)]
            slot_max_sessions = self.scenario_params['slot_max_sessions'][str(i)]
            
            # Since this agent only makes one decision, constraints on session count are primarily for the global aggregator.
            # We only need to ensure we pick *a* slot.
            
            scores.append((score, i))

        # Select the slot with the minimum score (best utility)
        best_score, best_slot_idx = min(scores, key=lambda x: x[0])
        
        return best_slot_idx

    def generate_recommendation(self) -> List[int]:
        """
        Generates the 7-day slot recommendation.
        """
        recommendations = []
        day_keys = list(self.days_data.keys())
        
        # To handle noise, we will calculate the average utility over a small perturbation, 
        # but for a deterministic policy, we rely on the core calculation unless specified otherwise.
        
        # We iterate through the 7 days in order.
        for i, day_key in enumerate(day_keys):
            day_data = self.days_data[day_key]
            
            # Note: The "Forecast note" suggests noise (±20%). A robust policy would iterate multiple times 
            # with noise applied and choose the most robust slot. For this deterministic setup, we use the nominal values.
            
            recommended_slot = self._determine_daily_slot(day_key, day_data)
            recommendations.append(recommended_slot)
            
        return recommendations

    def save_output(self, recommendations: List[int]):
        output_data = {
            "agent_id": self.agent_id,
            "scenario_id": self.data['scenario_id'],
            "recommendations": [int(slot) for slot in recommendations]
        }
        
        # Save global_policy_output.json
        with open('global_policy_output.json', 'w') as f:
            json.dump(output_data, f, indent=4)

def main():
    # Agent 3
    agent_id = 3
    policy = Policy(agent_id)
    
    recommendations = policy.generate_recommendation()
    
    # Save results
    policy.save_output(recommendations)

if __name__ == "__main__":
    main()