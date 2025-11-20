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
NEIGHBOR_PROFILES = {
    # Neighbor 2: Transformer headroom focus (Location 2)
    2: {'base_demand': [0.70, 1.00, 0.80, 0.50], 'preferred_slots': [1, 2], 'comfort_penalty': 0.14},
    # Neighbor 3: Nurse on central ridge (Location 3)
    3: {'base_demand': [0.60, 0.80, 0.90, 0.70], 'preferred_slots': [1, 3], 'comfort_penalty': 0.20},
}

# --- Helper Functions ---

def load_scenario_data(filename: str) -> Dict[str, Any]:
    """Loads scenario data from the specified JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def calculate_net_load(day_data: Dict[str, Any], slot_index: int, num_sessions: int) -> float:
    """
    Calculates the net load contribution based on demand, capacity, and sessions.
    Since the agent only controls its charging session size, we assume the charging
    rate is proportional to the number of sessions relative to capacity/slots, 
    but for simplicity in this heuristic context, we use the capacity allocated
    per session, subtracted from the baseline, to estimate congestion impact.
    
    Given this is Stage 3 (Collective), the goal is often to manage demand relative
    to baseline and capacity constraints.
    
    We assume a fixed charging rate per session needed to meet the base demand gap, 
    or simply use the capacity distributed across the session count for scoring.
    
    Let's simplify: The total load imposed by the agent is num_sessions * Avg_Charge_Rate.
    If we must meet the total demand, the session size is what matters for cost/carbon.
    We will use a standard rate proportional to capacity for simplicity if sessions are active.
    
    If we must provide a total energy amount (E_total), E_total = num_sessions * Rate * 1h.
    We aim to satisfy the load profile, so we assume the *total* required energy is met.
    The cost/carbon scores reflect the *time* of delivery.
    
    We will use the *total* capacity allocated to the agent in that slot as the load magnitude,
    normalized by slot min/max constraints, for scoring the congestion/capacity impact.
    """
    
    # For simplicity, assume the total energy required is fixed, and we are just choosing WHEN.
    # The actual load magnitude used for scoring congestion/capacity should relate to 
    # how much load this agent is imposing in that slot.
    
    # Since we don't know the total required energy, we use a proxy: the average capacity per session.
    # A session is 1 hour.
    
    # A common assumption in these models is that the agent needs to meet its BASE_DEMAND + a variable charge amount.
    # In this context, we evaluate the cost/carbon/spatial metrics based on the *time* slot chosen.
    
    # Heuristic Decision: The load magnitude for scoring is based on the maximum possible load if one session runs.
    # This proxies the congestion caused by running the charging session at that time.
    
    if num_sessions > 0:
        # Proxy load: How much energy is delivered in this slot? 
        # Let's assume max charge rate per active session is AGENT_CAPACITY / (Max possible sessions, e.g., 2)
        # If we assume the agent needs 5 kWh total across the week, this is complex.
        
        # Standard approach for time-based selection: The decision score depends only on the time factors (C, P, S).
        # The number of sessions (N_s) will act as a multiplier on the comfort/local penalty (GAMMA),
        # ensuring that more activity is penalized locally if constraints exist.
        
        # We calculate the *time cost* for one session, then weight by the actual sessions planned.
        
        # We calculate the expected baseline load for comparison against capacity limits (not strictly needed for scoring below, but good context)
        # baseline_load = day_data['Baseline load'][slot_index]
        
        # We use the agent's base demand *for this slot* as a component of congestion impact if active.
        base_demand_slot = AGENT_BASE_DEMAND[slot_index]
        
        # If the agent is active (num_sessions > 0), it imposes a congestion cost proportional to its activity level.
        # We use the number of sessions as the load multiplier for local penalties.
        return num_sessions * (base_demand_slot + 1.0) # +1.0 as a proxy for charging activity load
    return 0.0


def calculate_objective_score(
    day_index: int, 
    slot_index: int, 
    scenario: Dict[str, Any], 
    num_sessions: int,
    neighbor_data: Dict[int, Dict[str, Any]]
) -> float:
    """
    Calculates a composite score for choosing a specific slot on a specific day.
    Lower score is better.
    Score = alpha*Carbon + beta*Price + gamma*Spatial_Carbon_Penalty
    """
    
    day_key = list(scenario['days'].keys())[day_index]
    day_data = scenario['days'][day_key]
    
    # 1. Carbon Intensity (Global Goal: Minimize)
    # Use the scenario's specific carbon intensity for the day and slot.
    carbon_intensity = day_data['Carbon'][slot_index]
    
    # 2. Price (Local Goal: Minimize Cost)
    tariff = day_data['Tariff'][slot_index]
    
    # 3. Comfort / Spatial Carbon (Local/Neighbor Goal: Minimize congestion/thermal stress)
    spatial_carbon_key = f"{AGENT_LOCATION}: " + "; ".join(
        str(day_data['Spatial carbon'][loc]) for loc in range(1, len(scenario['days'][day_key]['Spatial carbon']))
    )
    
    # Extract the specific spatial carbon value for this agent's location (key=AGENT_LOCATION)
    spatial_carbon_values = day_data['Spatial carbon']
    
    # spatial_carbon_values is a dict mapping location string keys (e.g., '1') to carbon values.
    # We must parse the string format provided in the input to get the correct value structure.
    
    # Re-parsing the spatial carbon structure based on input definition:
    # Example: '1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ...'
    
    def parse_spatial_carbon(s_carbon_str: str, target_loc: int) -> float:
        try:
            parts = s_carbon_str.split(';')
            for part in parts:
                if part.strip().startswith(f"{target_loc}:"):
                    # Extract the string of values for this location (e.g., " 330, 520, 560, 610")
                    values_str = part.split(':')[1].strip()
                    slot_values = [float(v.strip()) for v in values_str.split(',')]
                    if 0 <= slot_index < len(slot_values):
                        return slot_values[slot_index]
            # Fallback if location not found (should not happen)
            return scenario['spatial_carbon'][slot_index] 
        except Exception:
            # Fallback to central grid carbon if parsing fails dramatically
            return scenario['carbon_intensity'][slot_index]

    spatial_carbon_local = parse_spatial_carbon(spatial_carbon_values, AGENT_LOCATION)
    
    # Neighbor Context: If neighbors prefer specific slots, deviating from them might increase collective stress 
    # unless those preferred slots align with the agent's low-cost/low-carbon choice.
    
    neighbor_coordination_penalty = 0.0
    
    # Battery engineer (Agent 1) should try to cooperate with congestion-aware neighbors (e.g., N2)
    # and nurses concerned about night shift reliability (N3).
    
    for nid, n_data in neighbor_data.items():
        if n_data['preferred_slots'][slot_index] == 1: # Simplified check: is this slot preferred by the neighbor?
            # If the slot is preferred by neighbors, running here might cause local congestion, 
            # UNLESS the agent's decision is *already* minimizing carbon/price heavily.
            # Since this agent prioritizes global goals (Carbon/Price), we penalize if we *clash* with a neighbor's preference
            # only if that neighbor explicitly signals congestion concern (like N2).
            
            # For a general coordination heuristic, we slightly increase the penalty if we are running during a slot 
            # where neighbors are showing activity preference, as this suggests collective load.
            neighbor_coordination_penalty += 0.1 * n_data['comfort_penalty'] 
            
    
    # 4. Compute Weighted Score
    # The agent wants low carbon (Alpha) and low price (Beta). 
    # It wants low spatial carbon/congestion (Gamma), weighted higher due to the high alpha (global focus).
    
    # The number of sessions modifies the local/comfort penalty (Gamma) strongly.
    # Base Score = alpha*C + beta*P + gamma*S
    base_score = (
        ALPHA * carbon_intensity + 
        BETA * tariff + 
        GAMMA * spatial_carbon_local
    )
    
    # Apply Session Multiplier to the local/comfort component (Gamma) and coordination penalty
    # If num_sessions is 0, the score should be effectively infinite (or very high) unless we are explicitly modeling 'not charging'.
    if num_sessions == 0:
        return 1e6 # Effectively disqualify slots with 0 sessions planned
        
    final_score = base_score + (num_sessions * GAMMA * 0.5 * neighbor_coordination_penalty)
    
    return final_score


def determine_session_count(day_index: int, slot_index: int, scenario: Dict[str, Any]) -> int:
    """
    Determines the required session count for this agent on this day/slot, 
    respecting min/max constraints, while aiming to satisfy baseline demand gap 
    (if known) and collective coordination mandates (e.g., transformer temps).
    
    Agent 1 (Battery Engineer) needs to meet its base demand profile + likely charge 
    a certain amount, constrained by capacity (6.8 kW total).
    
    Heuristic: We need to meet the total demand gap over the 4 slots. 
    Total required energy is roughly the sum of the differences between Capacity and Baseline Load.
    
    Total Required Energy Proxy (kWh): Sum over slots (Baseline - Capacity) if positive, adjusted by noise expectation.
    Since we don't have the target EV energy, we rely on the slot constraints (min/max sessions).
    
    Goal: Distribute sessions to meet overall demand needs (which often means running when C/P is low)
    while respecting local constraints (e.g., Day 2: balance temps).
    
    We prioritize filling slots that have lower carbon/price targets first, up to max sessions.
    We ensure min sessions are met first.
    
    Since this is Echo Stage 3 (Collective), we must look at constraints mentioned in the day descriptions.
    """
    day_key = list(scenario['days'].keys())[day_index]
    day_data = scenario['days'][day_key]
    
    min_s = scenario['slot_min_sessions'][slot_index]
    max_s = scenario['slot_max_sessions'][slot_index]
    
    # --- Day-Specific Coordination Logic (Derived from Forecast Notes) ---
    
    # Day 2 (Wind ramps mean slots 0 and 3 must balance transformer temps.)
    if 'Day 2' in day_key:
        if slot_index == 0 or slot_index == 3:
            # Must ensure activity here, likely high activity to manage temp swings
            return max_s 
        else:
            # Moderate activity expected elsewhere
            return min_s
            
    # Day 6 (Maintenance advisory caps the valley transformer; slot 2 is rationed.)
    if 'Day 6' in day_key:
        if slot_index == 2:
            # Rationed slot -> set to minimum possible session count
            return min_s
        # Other slots can be used to compensate
    
    # Day 4 (Neighborhood watch enforces staggered use before the late-event recharge.)
    if 'Day 4' in day_key:
        # Staggered use implies spreading sessions out. 
        # If we are slot 3 (late), we might reduce activity if slot 0/1/2 were used heavily.
        # Without knowing activity in other slots, we default to medium activity unless slot 3.
        if slot_index == 3:
            return min_s # Reduce late evening usage relative to max
        return max_s # Try to use early slots fully

    # Default Strategy: Meet minimum requirement, but fill up slots that are generally good (low carbon/price) 
    # up to max capacity, ensuring collective coverage if possible.
    
    # Since Agent 1 prioritizes global goals (Carbon/Price), we can use the *scenario average* 
    # carbon/price for this slot to decide if we should push for max sessions (if they are low).
    
    avg_carbon = scenario['carbon_intensity'][slot_index]
    avg_price = scenario['price'][slot_index]
    
    if avg_carbon < 500 and avg_price < 0.26:
        # Good period overall -> push for high activity if not overridden by day constraints
        return max_s
    elif avg_carbon > 600 or avg_price > 0.28:
        # Expensive/Dirty period -> stick to minimum necessary activity
        return min_s
    else:
        # Moderate -> meet minimum, perhaps one extra session if max allows
        return min(min_s + 1, max_s)


def recommend_slots(scenario: Dict[str, Any], neighbor_data: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Determines the recommended slot index (0-3) for each of the 7 days.
    """
    
    num_days = len(scenario['days'])
    recommendations = []
    
    # Pre-calculate neighbor preferences for coordination visibility
    # Since neighbors provided 'Ground truth min-cost slots by day', we use these as signals of *where they will be*.
    
    # Map neighbor ground truth to a daily slot preference indicator (1 if slot X is preferred today)
    neighbor_daily_preference = {nid: {d: [0] * len(scenario['slots']) for d in range(num_days)} for nid in neighbor_data}
    
    for nid, n_data in neighbor_data.items():
        gt_slots = n_data['ground_truth_min_cost_slots_by_day']
        for day_idx, slot_list in enumerate(gt_slots):
            for slot in slot_list:
                if 0 <= slot < len(scenario['slots']):
                    neighbor_daily_preference[nid][day_idx][slot] = 1
    
    
    for day_idx in range(num_days):
        best_score = float('inf')
        best_slot = -1
        
        # 1. Determine session counts for this day based on mandates/heuristics
        daily_sessions = {}
        for slot_idx in range(len(scenario['slots'])):
            daily_sessions[slot_idx] = determine_session_count(day_idx, slot_idx, scenario)
            
        # 2. Reconstruct neighbor data structure for scoring function, using today's context
        current_neighbor_context = {}
        for nid, n_data in neighbor_data.items():
            # Merge static profile info with dynamic session context (if available, otherwise use default preference signal)
            context = n_data.copy()
            # Add the current day's activity signal (which slot is preferred/active)
            context['current_session_plan'] = neighbor_daily_preference[nid][day_idx]
            current_neighbor_context[nid] = context

        
        # 3. Score each slot based on the determined session count
        for slot_idx in range(len(scenario['slots'])):
            
            num_s = daily_sessions[slot_idx]
            
            # Only score slots where charging is actually planned (num_s >= min_s >= 1 usually)
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
            # Fallback: If no slot scored well (e.g., all failed session requirements), pick the lowest price slot.
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
        print(f"Error: {SCENARIO_FILE} not found.")
        return

    # Augment neighbor data with their ground truth session plans for coordination awareness
    # (We must manually insert the ground truth slots into the structure expected by the scoring function)
    
    augmented_neighbors = {}
    
    # Neighbor 2 (Location 2)
    N2_GT = [1, 2, 0, 1, 2, 0, 1] # Indices for Day 1 to Day 7
    augmented_neighbors[2] = NEIGHBOR_PROFILES[2].copy()
    augmented_neighbors[2]['ground_truth_min_cost_slots_by_day'] = N2_GT
    
    # Neighbor 3 (Location 3)
    N3_GT = [2, 0, 1, 3, 0, 1, 2] # Indices for Day 1 to Day 7
    augmented_neighbors[3] = NEIGHBOR_PROFILES[3].copy()
    augmented_neighbors[3]['ground_truth_min_cost_slots_by_day'] = N3_GT

    # 2. Decide Slot Recommendation
    recommended_slots = recommend_slots(scenario, augmented_neighbors)
    
    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": "Agent 1",
        "scenario_id": scenario['scenario_id'],
        "recommendations": [
            {"day": i + 1, "slot_index": rec} for i, rec in enumerate(recommended_slots)
        ]
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
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
NEIGHBOR_PROFILES = {
    # Neighbor 2: Transformer headroom focus (Location 2)
    2: {'base_demand': [0.70, 1.00, 0.80, 0.50], 'preferred_slots': [1, 2], 'comfort_penalty': 0.14},
    # Neighbor 3: Nurse on central ridge (Location 3)
    3: {'base_demand': [0.60, 0.80, 0.90, 0.70], 'preferred_slots': [1, 3], 'comfort_penalty': 0.20},
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
                # Extract the string of values for this location (e.g., " 330, 520, 560, 610")
                values_str = part.split(':')[1].strip()
                slot_values = [float(v.strip()) for v in values_str.split(',')]
                if 0 <= slot_index < len(slot_values):
                    return slot_values[slot_index]
        # Fallback to the scenario average if parsing fails or location is not found
        return 1000.0 # High penalty if location data is missing
    except Exception:
        # Fallback to a high value if parsing fails dramatically
        return 1000.0 

def calculate_objective_score(
    day_index: int, 
    slot_index: int, 
    scenario: Dict[str, Any], 
    num_sessions: int,
    neighbor_data: Dict[int, Dict[str, Any]]
) -> float:
    """
    Calculates a composite score for choosing a specific slot on a specific day.
    Lower score is better.
    Score = alpha*Carbon + beta*Price + gamma*Spatial_Carbon_Penalty + Coordination_Penalty
    """
    
    day_key = list(scenario['days'].keys())[day_index]
    day_data = scenario['days'][day_key]
    
    # 1. Carbon Intensity (Global Goal: Minimize)
    carbon_intensity = day_data['Carbon'][slot_index]
    
    # 2. Price (Local Goal: Minimize Cost)
    tariff = day_data['Tariff'][slot_index]
    
    # 3. Spatial Carbon / Congestion (Local Goal: Minimize local stress)
    spatial_carbon_values_str = day_data['Spatial carbon']
    spatial_carbon_local = parse_spatial_carbon(spatial_carbon_values_str, AGENT_LOCATION, slot_index)
    
    # 4. Neighbor Coordination Penalty (Attempt to avoid clashes with neighbors' known critical times)
    neighbor_coordination_penalty = 0.0
    
    for nid, n_data in neighbor_data.items():
        # Check if this slot is currently active/preferred according to the neighbor's stated ground truth for today
        current_plan = n_data.get('current_session_plan', [0]*len(scenario['slots']))
        
        if current_plan[slot_index] == 1:
            # A neighbor is active here. We slightly penalize if we clash, unless their comfort penalty is low (suggesting low urgency)
            # We use N2 (transformer analyst) signal more heavily if they are active.
            if nid == 2:
                 neighbor_coordination_penalty += 1.5 * n_data['comfort_penalty'] 
            else:
                 neighbor_coordination_penalty += 0.5 * n_data['comfort_penalty']
            
    
    # 5. Compute Weighted Score
    if num_sessions == 0:
        return 1e9 # Effectively disqualify slots where no charging is planned
        
    # Base Score weights global (C, P) and local congestion (S)
    base_score = (
        ALPHA * carbon_intensity + 
        BETA * tariff + 
        GAMMA * spatial_carbon_local
    )
    
    # The local cost (congestion + coordination) is amplified by the actual activity level (num_sessions)
    activity_weighted_penalty = (
        num_sessions * GAMMA * 0.5 * neighbor_coordination_penalty 
        + num_sessions * 1.0 # Base penalty for imposing load
    )
    
    final_score = base_score + activity_weighted_penalty
    
    return final_score


def determine_session_count(day_index: int, slot_index: int, scenario: Dict[str, Any]) -> int:
    """
    Determines the required session count for this agent on this day/slot, 
    based on day-specific mandates and general efficiency goals.
    """
    day_key = list(scenario['days'].keys())[day_index]
    day_data = scenario['days'][day_key]
    
    min_s = scenario['slot_min_sessions'][slot_index]
    max_s = scenario['slot_max_sessions'][slot_index]
    
    # --- Day-Specific Coordination Logic (Mandates) ---
    
    # Day 2 (Wind ramps mean slots 0 and 3 must balance transformer temps.)
    if 'Day 2' in day_key:
        if slot_index == 0 or slot_index == 3:
            # Must ensure activity here to manage temps
            return max_s 
        else:
            # Moderate activity expected elsewhere, meet minimums
            return min_s
            
    # Day 6 (Maintenance advisory caps the valley transformer; slot 2 is rationed.)
    if 'Day 6' in day_key:
        if slot_index == 2:
            # Rationed slot -> set to minimum possible session count
            return min_s
        # Other slots can be used more aggressively to compensate, up to max
        return max_s
    
    # Day 4 (Neighborhood watch enforces staggered use before the late-event recharge.)
    if 'Day 4' in day_key:
        # Staggered use implies spreading sessions out. Avoid high activity in the last slot if possible.
        if slot_index == 3:
            return min_s # Reduce late evening usage relative to max
        return max_s # Use early/mid slots fully

    # --- Default Strategy: Cost/Carbon Optimization ---
    
    avg_carbon = scenario['carbon_intensity'][slot_index]
    avg_price = scenario['price'][slot_index]
    
    # Battery engineer prioritizes low carbon/price globally
    if avg_carbon < 490 and avg_price < 0.25:
        # Excellent period -> push for high activity
        return max_s
    elif avg_carbon > 550 or avg_price > 0.29:
        # Expensive/Dirty period -> stick to minimum necessary activity
        return min_s
    else:
        # Moderate -> aim for one session above minimum if possible
        return min(min_s + 1, max_s)


def recommend_slots(scenario: Dict[str, Any], neighbor_data: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Determines the recommended slot index (0-3) for each of the 7 days.
    """
    
    num_days = len(scenario['days'])
    recommendations = []
    
    # Map neighbor ground truth to a daily slot preference indicator (1 if slot X is used by neighbor)
    neighbor_daily_preference = {nid: {d: [0] * len(scenario['slots']) for d in range(num_days)} for nid in neighbor_data}
    
    for nid, n_data in neighbor_data.items():
        gt_slots = n_data['ground_truth_min_cost_slots_by_day']
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
            
            if num_s >= scenario['slot_min_sessions'][slot_idx]:
                
                score = calculate_objective_score(
                    day_index=day_idx, 
                    slot_index=slot_idx, 
                    scenario=scenario, 
                    num_sessions=num_s,
                    neighbor_data=current_neighbor_context
                )
                
                # Tie-breaking: Prefer lower index (earlier slot) if scores are equal
                if score < best_score:
                    best_score = score
                    best_slot = slot_idx
        
        if best_slot == -1:
            # Fallback: Pick the lowest price slot if constraints failed optimization
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
        # In a real environment, this might raise an error or exit. Here, we ensure compliance.
        return

    # Augment neighbor data with their ground truth session plans (Crucial for Coordination Logic)
    augmented_neighbors = {}
    
    # Neighbor 2 (Location 2) - Ground truth min-cost slots: Day 1:[1]; D2:[2]; D3:[0]; D4:[1]; D5:[2]; D6:[0]; D7:[1]
    N2_GT = [1, 2, 0, 1, 2, 0, 1] 
    augmented_neighbors[2] = NEIGHBOR_PROFILES[2].copy()
    augmented_neighbors[2]['ground_truth_min_cost_slots_by_day'] = N2_GT
    
    # Neighbor 3 (Location 3) - Ground truth min-cost slots: Day 1:[2]; D2:[0]; D3:[1]; D4:[3]; D5:[0]; D6:[1]; D7:[2]
    N3_GT = [2, 0, 1, 3, 0, 1, 2] 
    augmented_neighbors[3] = NEIGHBOR_PROFILES[3].copy()
    augmented_neighbors[3]['ground_truth_min_cost_slots_by_day'] = N3_GT

    # 2. Decide Slot Recommendation
    recommended_slots = recommend_slots(scenario, augmented_neighbors)
    
    # 3. Write global_policy_output.json
    output_data = {
        "agent_id": "Agent 1",
        "scenario_id": scenario.get('scenario_id', 'unknown_scenario'),
        "recommendations": [
            {"day": i + 1, "slot_index": rec} for i, rec in enumerate(recommended_slots)
        ]
    }
    
    with open('global_policy_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()