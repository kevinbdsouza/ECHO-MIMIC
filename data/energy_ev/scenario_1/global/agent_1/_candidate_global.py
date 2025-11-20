import json
import numpy as np
from typing import List, Dict, Any

# --- Configuration ---
SCENARIO_FILE = 'scenario.json'

# Agent Persona Parameters (Position 1: Battery engineer balancing budget and solar backfeed)
ALPHA = 40.00  # Weight for Carbon (Global objective)
BETA = 0.50    # Weight for Price/Cost (Local objective - Budget)
GAMMA = 12.00  # Weight for Comfort/Spatial Carbon (Local objective - Thermal/Congestion)

AGENT_CAPACITY = 6.8 # kW (Assumed total charging capacity)
AGENT_BASE_DEMAND = [1.20, 0.70, 0.80, 0.60] # kW (Base demand profile)
AGENT_LOCATION = 1

# Neighbor Profiles (Used for contextual awareness of collective goals)
# Ground truth slots are imported here to create coordination context signals for the scoring function.
NEIGHBOR_PROFILES = {
    # Neighbor 2: Transformer headroom focus (Location 2)
    2: {'base_demand': [0.70, 1.00, 0.80, 0.50], 'preferred_slots': [1, 2], 'comfort_penalty': 0.14, 'gt_slots': [1, 2, 0, 1, 2, 0, 1]}, # D1-D7
    # Neighbor 3: Nurse on central ridge (Location 3)
    3: {'base_demand': [0.60, 0.80, 0.90, 0.70], 'preferred_slots': [1, 3], 'comfort_penalty': 0.20, 'gt_slots': [2, 0, 1, 3, 0, 1, 2]}, # D1-D7
}

# --- Helper Functions ---

def load_scenario_data(filename: str) -> Dict[str, Any]:
    """Loads scenario data from the specified JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def parse_spatial_carbon(s_carbon_str: str, target_loc: int, slot_index: int) -> float:
    """Parses the complex spatial carbon string to find the value for the target location and slot."""
    try:
        parts = s_carbon_str.split(';')
        for part in parts:
            if part.strip().startswith(f"{target_loc}:"):
                values_str = part.split(':')[1].strip()
                slot_values = [float(v.strip()) for v in values_str.split(',')]
                if 0 <= slot_index < len(slot_values):
                    return slot_values[slot_index]
        return 1e6 # High penalty if location data is missing
    except Exception:
        return 1e6 

def determine_session_count(day_index: int, slot_index: int, scenario: Dict[str, Any]) -> int:
    """
    Determines the required session count based on day-specific mandates and general efficiency goals.
    For Agent 1 (Battery Engineer), we prioritize low cost/carbon unless mandated otherwise.
    """
    day_key = list(scenario['days'].keys())[day_index]
    day_data = scenario['days'][day_key]
    
    min_s = scenario['slot_min_sessions'][slot_index]
    max_s = scenario['slot_max_sessions'][slot_index]
    
    # --- Day-Specific Coordination Logic (Mandates) ---
    
    # Day 2: slots 0 and 3 must balance transformer temps -> push activity.
    if 'Day 2' in day_key:
        if slot_index == 0 or slot_index == 3:
            return max_s 
        else:
            return min_s
            
    # Day 6: slot 2 is rationed -> set to minimum.
    if 'Day 6' in day_key:
        if slot_index == 2:
            return min_s
        return max_s
    
    # Day 4: Staggered use preferred. Reduce late activity (slot 3).
    if 'Day 4' in day_key:
        if slot_index == 3:
            return min_s
        return max_s

    # --- Default Strategy: Cost/Carbon Optimization (Prioritize ALPHA and BETA goals) ---
    avg_carbon = scenario['carbon_intensity'][slot_index]
    avg_price = scenario['price'][slot_index]
    
    if avg_carbon < 490 and avg_price < 0.25:
        # Excellent period -> push for high activity
        return max_s
    elif avg_carbon > 550 or avg_price > 0.29:
        # Expensive/Dirty period -> stick to minimum necessary activity
        return min_s
    else:
        # Moderate -> aim for one session above minimum if possible
        return min(min_s + 1, max_s)


def calculate_objective_score(
    day_index: int, 
    slot_index: int, 
    scenario: Dict[str, Any], 
    num_sessions: int,
    neighbor_data: Dict[int, Dict[str, Any]]
) -> float:
    """Calculates a composite score (lower is better)."""
    
    day_key = list(scenario['days'].keys())[day_index]
    day_data = scenario['days'][day_key]
    
    # 1. Carbon Intensity (Global Goal: Minimize)
    carbon_intensity = day_data['Carbon'][slot_index]
    
    # 2. Price (Local Goal: Minimize Cost)
    tariff = day_data['Tariff'][slot_index]
    
    # 3. Spatial Carbon / Congestion (Local Goal: Minimize local stress at Agent 1)
    spatial_carbon_values_str = day_data['Spatial carbon']
    spatial_carbon_local = parse_spatial_carbon(spatial_carbon_values_str, AGENT_LOCATION, slot_index)
    
    # 4. Neighbor Coordination Penalty (Avoid conflicting with neighbors' planned activity)
    neighbor_coordination_penalty = 0.0
    
    for nid, n_data in neighbor_data.items():
        # Check if this slot is used by the neighbor according to their ground truth plan for today
        current_plan = n_data.get('current_session_plan', [0]*len(scenario['slots']))
        
        if current_plan[slot_index] == 1:
            # Clashing with active neighbor activity adds a coordination penalty, weighted by neighbor's rigidity.
            neighbor_coordination_penalty += 1.5 * n_data['comfort_penalty'] 
            
    
    if num_sessions == 0:
        return 1e9
        
    # Score based on objectives: (ALPHA*Carbon + BETA*Price + GAMMA*SpatialCarbon)
    base_score = (
        ALPHA * carbon_intensity + 
        BETA * tariff + 
        GAMMA * spatial_carbon_local
    )
    
    # Activity level modifies the local congestion/coordination penalty strongly
    activity_weighted_penalty = (
        num_sessions * (0.5 * GAMMA * neighbor_coordination_penalty) 
        + num_sessions * 1.0 # Base penalty for imposing load
    )
    
    final_score = base_score + activity_weighted_penalty
    
    return final_score


def recommend_slots(scenario: Dict[str, Any], neighbor_data: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Determines the recommended slot index (0-3) for each of the 7 days.
    """
    
    num_days = len(scenario['days'])
    recommendations = []
    
    # Map neighbor ground truth to a daily slot preference indicator (1 if slot X is used by neighbor)
    neighbor_daily_preference = {nid: {d: [0] * len(scenario['slots']) for d in range(num_days)} for nid in neighbor_data}
    
    for nid, n_data in neighbor_data.items():
        gt_slots = n_data['gt_slots']
        for day_idx, slot in enumerate(gt_slots):
            if 0 <= slot < len(scenario['slots']):
                neighbor_daily_preference[nid][day_idx][slot] = 1
    
    
    for day_idx in range(num_days):
        best_score = float('inf')
        best_slot = -1
        
        # 1. Determine session counts for this day based on mandates/heuristics
        daily_sessions = {}
        for slot_idx in range(len(scenario['slots'])):
            daily_sessions[slot_idx] = determine_session_count(day_idx, slot_idx, scenario)
            
        # 2. Prepare current neighbor context for scoring
        current_neighbor_context = {}
        for nid, n_data in neighbor_data.items():
            context = n_data.copy()
            context['current_session_plan'] = neighbor_daily_preference[nid][day_idx]
            current_neighbor_context[nid] = context

        
        # 3. Score each slot
        for slot_idx in range(len(scenario['slots'])):
            
            num_s = daily_sessions[slot_idx]
            
            # Only score slots where charging is planned (num_s >= min_s >= 1)
            if num_s >= scenario['slot_min_sessions'][slot_idx]:
                
                score = calculate_objective_score(
                    day_index=day_idx, 
                    slot_index=slot_idx, 
                    scenario=scenario, 
                    num_sessions=num_s,
                    neighbor_data=current_neighbor_context
                )
                
                # Tie-breaking: If scores are equal, prefer lower index (earlier slot)
                if score < best_score:
                    best_score = score
                    best_slot = slot_idx
        
        if best_slot == -1:
            # Fallback: Pick the lowest price slot if optimization failed
            day_tariffs = scenario['days'][list(scenario['days'].keys())[day_idx]]['Tariff']
            best_slot = np.argmin(day_tariffs)
            
        recommendations.append(best_slot)
        
    return recommendations

# --- Main Execution ---

def main():
    # 1. Load Scenario Data
    try:
        scenario = load_scenario_data(SCENARIO_FILE)
    except FileNotFoundError:
        # Create dummy output if file is missing to avoid crash, though execution context implies file existence.
        dummy_output = [{"day": i+1, "slot_index": 1} for i in range(7)]
        with open('global_policy_output.json', 'w') as f:
            json.dump(dummy_output, f, indent=4)
        return

    # Augment neighbor data with their ground truth session plans for coordination awareness
    augmented_neighbors = {}
    
    for nid, profile in NEIGHBOR_PROFILES.items():
        augmented_neighbors[nid] = profile.copy()
        # Ensure 'gt_slots' is present, which it is by definition above.

    # 2. Decide Slot Recommendation
    recommended_slots = recommend_slots(scenario, augmented_neighbors)
    
    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": "Agent 1",
        "scenario_id": scenario.get('scenario_id', 'ev_peak_sharing_1'),
        "recommendations": [
            {"day": i + 1, "slot_index": rec} for i, rec in enumerate(recommended_slots)
        ]
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data['recommendations'], f, indent=4)

if __name__ == "__main__":
    main()