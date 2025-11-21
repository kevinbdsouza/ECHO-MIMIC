import json
import os
from typing import List, Dict, Any

# Define constants based on the scenario description
SLOTS_PER_DAY = 4
DAYS_IN_WEEK = 7

# Agent specific parameters
BASE_DEMAND = [0.60, 0.80, 0.90, 0.70]
ALPHA = 40.00
BETA = 0.50
GAMMA = 12.00
CAPACITY = 6.8
LOCATION = 3

# --- Helper Functions ---

def load_scenario_data(base_path: str = ".") -> Dict[str, Any]:
    """Loads the scenario configuration from scenario.json."""
    file_path = os.path.join(base_path, "scenario.json")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: scenario.json not found at {file_path}")
        # Return a structure that allows testing if actual loading fails
        return {
            "slots": {str(i): f"{i*10+19}-{i*10+20}" for i in range(SLOTS_PER_DAY)},
            "price": [0.23, 0.24, 0.27, 0.30],
            "carbon_intensity": [700, 480, 500, 750],
            "capacity": 6.8,
            "baseline_load": [5.2, 5.0, 4.9, 6.5],
            "slot_min_sessions": {str(i): 1 for i in range(SLOTS_PER_DAY)},
            "slot_max_sessions": {str(i): 2 for i in range(SLOTS_PER_DAY)},
            "spatial_carbon": {
                "1": "440, 460, 490, 604", "2": "483, 431, 471, 600", "3": "503, 473, 471, 577",
                "4": "617, 549, 479, 363", "5": "411, 376, 554, 623"
            },
            "days": {
                "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5],
                          "Spatial carbon": {"1": "330, 520, 560, 610", "2": "550, 340, 520, 600", "3": "590, 520, 340, 630", "4": "620, 560, 500, 330", "5": "360, 380, 560, 620"}},
                "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6],
                          "Spatial carbon": {"1": "510, 330, 550, 600", "2": "540, 500, 320, 610", "3": "310, 520, 550, 630", "4": "620, 540, 500, 340", "5": "320, 410, 560, 640"}},
                # ... add placeholders for all 7 days for robustness
            },
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA
        }

def parse_spatial_carbon(sc_data: Dict[str, str], agent_loc: int) -> List[float]:
    """Parses spatial carbon data for the agent's location."""
    # Format: 1: 440, 460, 490, 604
    sc_str = sc_data.get(str(agent_loc))
    if sc_str:
        try:
            return [float(x.strip()) for x in sc_str.split(',')]
        except (ValueError, AttributeError):
            pass
    # Fallback if parsing fails
    return [500.0] * SLOTS_PER_DAY

def get_neighbor_info(scenario_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Extracts relevant neighbor information."""
    neighbors = {}
    if 'neighbor_examples' in scenario_data:
        for neighbor_name, data in scenario_data['neighbor_examples'].items():
            # Crude extraction of location based on name/position if location isn't explicitly given
            try:
                # Example: Neighbor 2, Position 2 -> Location 2
                loc_str = neighbor_name.split(' ')[1]
                loc = int(loc_str)
            except (IndexError, ValueError):
                # Fallback if naming convention is unexpected
                loc = -1 
            
            neighbors[loc] = {
                'base_demand': data.get('Base demand', [0.0]*SLOTS_PER_DAY),
                'preferred_slots': data.get('Preferred slots', []),
                'comfort_penalty': data.get('Comfort penalty', 0.1)
            }
    return neighbors

def calculate_cost(usage: List[float], day_data: Dict[str, Any], neighbor_sc: List[float], agent_loc: int) -> float:
    """
    Calculates the weighted cost (Carbon + Price + Spatial Carbon + Transformer Congestion).
    We prioritize minimizing carbon/price (Global) while respecting local comfort/capacity (Local).
    In Collective stage, the primary focus shifts to minimizing the sum of Price and Carbon,
    scaled by alpha/beta, and applying local constraints (gamma).
    """
    
    tariffs = day_data['Tariff']
    carbon_intensities = day_data['Carbon']
    baseline_load = day_data['Baseline load']
    
    total_cost = 0.0
    
    # 1. Carbon/Price Cost (Global Objective: Heavily weighted by alpha/beta)
    for t in range(SLOTS_PER_DAY):
        # Use Carbon (weighted by alpha) + Price (weighted by beta)
        cost_t = (ALPHA * carbon_intensities[t]) + (BETA * tariffs[t])
        total_cost += cost_t * usage[t]

    # 2. Spatial Carbon (Coordination/Local Factor: Penalize using slots where neighbors are carbon-heavy)
    # Since this agent is in Location 3, we use its spatial carbon forecast.
    # We penalize usage if the local grid carbon (neighbor_sc) is high.
    for t in range(SLOTS_PER_DAY):
        spatial_penalty = GAMMA * neighbor_sc[t] * usage[t]
        total_cost += spatial_penalty

    # 3. Baseline Load / Capacity Constraint (Implicitly handled by usage limits, but can be used as a penalty)
    # For simplicity in this stage, we rely on explicit bounds for transformer limits, 
    # but we can add a mild penalty if usage significantly exceeds the baseline load significantly
    # beyond the base demand, suggesting congestion.
    load_penalty = 0.0
    for t in range(SLOTS_PER_DAY):
        if usage[t] > BASE_DEMAND[t]:
             # Penalty proportional to how much load exceeds baseline, scaled by a small factor relative to alpha/beta
            load_penalty += 0.1 * (usage[t] - BASE_DEMAND[t]) * usage[t]
            
    total_cost += load_penalty
    
    return total_cost

def calculate_comfort_penalty(usage: List[float], neighbor_info: Dict[int, Dict]) -> float:
    """
    Calculates a penalty based on deviating from observed neighbor behavior, 
    especially if neighbors show strong consensus or if comfort constraints are explicitly defined.
    
    As a night-shift nurse (base demand heavily skewed towards later slots), 
    the agent prioritizes its own demand profile but needs to adjust for neighbors.
    
    Neighbors:
    - Neighbor 2 (Loc 2, Feeder Analyst): Prefers slots 1, 2. Heavily uses slot 2 on days 2, 6 (wind ramp/rationing).
    - Neighbor 5 (Loc 5, Graduate): Prefers slots 0, 1. Heavily uses slot 0 on most days.
    
    Coordination Strategy: Since this agent (Loc 3) is focused on balancing its high late-night demand (0.90 in slot 2),
    it should try to shift load *away* from the preferred slots of neighbors if those slots are low-carbon, 
    or shift load *towards* slots where neighbors are absent, unless global optimization dictates otherwise.
    Given the agent's high base demand in slot 2 (0.90), it inherently conflicts with Neighbor 2.
    
    We will implement a soft preference to *avoid* heavily loading slots where neighbors are strongly present, 
    unless global costs are very low there.
    """
    comfort_cost = 0.0
    
    # Neighbor 2 (Feeder Analyst, Loc 2) prefers slots 1, 2.
    # Neighbor 5 (Graduate, Loc 5) prefers slots 0, 1.
    
    # Agent 3's preferred slots based on base demand: [2] > [1] > [0, 3]
    
    # Strategy: Defer to neighbor usage if the usage is low (meaning they are allowing space)
    # or shift away if they are using high (meaning congestion risk).
    
    # For simplicity in Stage 3 (Collective), we use a penalty based on general proximity to neighbors' *preferred* behavior,
    # weighted by the neighbor's comfort penalty (indicating how fixed their schedule is).
    
    # Neighbor 2 (High commitment to 1, 2)
    n2_info = neighbor_info.get(2, {})
    if n2_info:
        for t in n2_info.get('preferred_slots', []):
            # If Agent 3 uses significantly more than Neighbor 2 did on their 'off' days, apply penalty.
            # We don't have neighbor usage history for the *current* week, only ground truth examples.
            # We will use the neighbor's base comfort penalty as a proxy for how much we should *avoid* conflicting with their preference.
            comfort_cost += n2_info.get('comfort_penalty', 0.1) * usage[t] * 0.5 # Soft penalty for using their preferred slots
            
    # Neighbor 5 (High commitment to 0, 1)
    n5_info = neighbor_info.get(5, {})
    if n5_info:
        for t in n5_info.get('preferred_slots', []):
            comfort_cost += n5_info.get('comfort_penalty', 0.1) * usage[t] * 0.5

    # Agent's Own Comfort: Prioritize usage according to BASE_DEMAND ranking (Slot 2 > 1 > 0, 3)
    # If usage is significantly lower than base demand in high-preference slots, add a penalty.
    own_comfort_penalty = 0.0
    for t in range(SLOTS_PER_DAY):
        # If usage is much lower than expected base demand (e.g., < 50% of base demand)
        if usage[t] < 0.5 * BASE_DEMAND[t]:
            own_comfort_penalty += (BASE_DEMAND[t] - usage[t]) * 0.1 
            
    return comfort_cost + own_comfort_penalty

def generate_usage_vector(day_index: int, scenario_data: Dict[str, Any], neighbor_info: Dict[int, Dict], spatial_carbon_forecast: List[float]) -> List[float]:
    """
    Generates the usage vector for a specific day by optimizing the trade-off 
    between global cost minimization (Carbon/Price) and local constraints/comfort.
    
    Since we cannot run a full non-linear optimization here, we use a heuristic 
    that targets the lowest cost slots but modulates based on the base demand profile 
    and neighbor presence.
    """
    day_key_prefix = "Day "
    day_key = [k for k in scenario_data['days'].keys() if k.startswith(f"{day_key_prefix}{day_index + 1}")][0]
    day_data = scenario_data['days'][day_key]

    # Parse dynamic data for the day
    day_tariffs = day_data['Tariff']
    day_carbon = day_data['Carbon']
    day_base_load = day_data['Baseline load']
    
    min_sessions = [scenario_data['slot_min_sessions'][str(i)] for i in range(SLOTS_PER_DAY)]
    max_sessions = [scenario_data['slot_max_sessions'][str(i)] for i in range(SLOTS_PER_DAY)]

    # 1. Initial Heuristic: Calculate Base Score (favoring low carbon/price)
    # We use the global cost function components for ranking, excluding the variable usage term initially.
    # Since we are in Stage 3 (Collective), we prioritize Global Cost minimization (Carbon + Price).
    
    base_scores = []
    for t in range(SLOTS_PER_DAY):
        # Score reflects the *cost per unit* of energy
        score = (ALPHA * day_carbon[t]) + (BETA * day_tariffs[t]) + (GAMMA * spatial_carbon_forecast[t])
        base_scores.append(score)
        
    # Create a list of (score, slot_index) tuples
    slot_ranking = sorted([(base_scores[t], t) for t in range(SLOTS_PER_DAY)])
    
    # 2. Initial Allocation based on Ranking and Agent Profile
    usage = [0.0] * SLOTS_PER_DAY
    
    # Calculate total required energy based on base demand (normalized to capacity)
    # In this setup, usage is a session fraction (0-1). We assume base demand implies a required usage level.
    
    # Determine the required total normalized energy based on the agent's base demand structure.
    # We need to satisfy the agent's profile structure while hitting the best slots.
    
    # Scale factor: How much energy (as a fraction of total possible, 4 slots * 1.0 usage) should be shifted?
    # For simplicity, assume the required fulfillment level (R) is based on the average of the base demand vector, normalized.
    required_fulfillment = sum(BASE_DEMAND) / (SLOTS_PER_DAY * 1.0) 
    
    # Determine the total usage volume needed to satisfy the agent's *shape*, 
    # while ensuring minimum sessions are met.
    
    # Start by setting minimums
    for t in range(SLOTS_PER_DAY):
        usage[t] = min_sessions[t] * (1.0 / max_sessions[t]) # Minimum load fraction (e.g., 1/2 = 0.5 if min=1, max=2)
        # For simplicity matching typical usage patterns where 1.0 means full session, let's enforce min session count in terms of usage volume (0/1)
        # Given usage is 0 to 1, we interpret min/max sessions as constraints on *how many* neighbors run, 
        # not directly on this agent's usage fraction (U_i). Since U_i is the agent's usage fraction, 
        # we treat min/max sessions as constraints on the number of agents, which we ignore, 
        # and rely only on min/max usage of 0/1, enforced by the slot constraints below.
        
        # Let's enforce the agent's base demand profile shape initially, then optimize the volume.
        usage[t] = BASE_DEMAND[t] / 1.0 # Start at base demand level
        
    
    # 3. Optimization Loop (Iterative adjustment based on cost and constraints)
    
    MAX_ITER = 50
    LEARNING_RATE = 0.05
    
    current_usage = [min(1.0, max(0.0, u)) for u in usage] # Clamp to [0, 1]

    for i in range(MAX_ITER):
        
        # Calculate cost gradient (heuristic approximation)
        # Change usage in the best slot slightly, and penalize usage in the worst slot slightly.
        
        delta_usage = [0.0] * SLOTS_PER_DAY
        
        # Prioritize pushing usage towards the lowest cost slots, up to capacity/max usage of 1.0
        best_slot = slot_ranking[0][1]
        worst_slot = slot_ranking[-1][1]

        # Check if we are deviating too much from the desired shape (BASE_DEMAND)
        shape_deviation = BASE_DEMAND[best_slot] - current_usage[best_slot]
        
        # If the best slot is currently under-utilized relative to the agent's expected shape, increase it.
        if shape_deviation > 0.1:
            delta_usage[best_slot] += LEARNING_RATE * shape_deviation
        
        # If the worst slot is over-utilized relative to the agent's expected shape, decrease it.
        shape_deviation_worst = current_usage[worst_slot] - BASE_DEMAND[worst_slot]
        if shape_deviation_worst > 0.1:
            delta_usage[worst_slot] -= LEARNING_RATE * shape_deviation_worst
            
        # Apply a small global shift based on carbon/price gradient, favoring low cost slots overall
        for t in range(SLOTS_PER_DAY):
            if base_scores[t] < sum(base_scores) / SLOTS_PER_DAY:
                # Shift slightly towards lower cost slots if they aren't already high
                if current_usage[t] < BASE_DEMAND[t] + 0.1:
                    delta_usage[t] += LEARNING_RATE * 0.01
            else:
                # Shift slightly away from high cost slots if they aren't already low
                if current_usage[t] > BASE_DEMAND[t] - 0.1:
                    delta_usage[t] -= LEARNING_RATE * 0.01


        # Apply changes
        next_usage = [current_usage[t] + delta_usage[t] for t in range(SLOTS_PER_DAY)]
        
        # Enforce hard constraints: [0, 1] and Min/Max Sessions (interpreted as usage bounds)
        # Since min_sessions/max_sessions are usually related to the *number* of agents, 
        # we primarily use the [0, 1] bound and the agent's desired profile shape via BASE_DEMAND/Comfort.
        
        new_usage = []
        for t in range(SLOTS_PER_DAY):
            u = next_usage[t]
            
            # Hard clamp to physical limits
            u = max(0.0, min(1.0, u))
            
            # Adjust based on minimum required usage (if we treat 1 session as 1.0 usage)
            # We must meet the minimum session requirement. If min=1, max=2, we must be >= 0.5 usage?
            # Given the uncertainty, we stick to the common interpretation for this stage: 
            # usage [0, 1] is the fraction of time the agent is active, and we enforce agent-specific baseline first.
            
            # Re-apply soft constraint: Don't deviate too much from the baseline shape unless global cost is extreme
            # If the cost is very high (e.g., carbon > 700 or price > 0.30), aggressively shift away from that slot, 
            # even if it means violating the BASE_DEMAND profile slightly.
            
            global_cost_threshold = (ALPHA * 600) + (BETA * 0.25) # High cost reference
            
            if base_scores[t] > global_cost_threshold:
                # Push usage down aggressively if the slot is costly
                u = max(0.0, u * 0.8) 
            
            new_usage.append(u)

        # Check convergence (simple total change threshold)
        if sum(abs(new_usage[t] - current_usage[t]) for t in range(SLOTS_PER_DAY)) < 0.001:
            break
            
        current_usage = new_usage

    # Final adjustment: Ensure minimum activity if the optimizer drove usage too low, 
    # prioritizing the slots that are *closest* to the original base demand profile among the low-cost options.
    
    final_usage = [max(0.0, u) for u in current_usage] # Final clamp
    
    # Post-process: Check comfort penalty vs. global cost benefit
    # If the lowest cost slot forces usage into Agent 3's low-preference time (e.g., slot 0 or 3), 
    # but the comfort penalty is high, we might slightly reduce usage there and increase usage in a slightly more expensive slot (like slot 1).
    
    # Since the iterative process already incorporated cost and the soft comfort factor indirectly via shape preference,
    # we stick to the converged result, ensuring it meets the *minimum* required sessions (if defined as volume).
    
    # Given the ambiguity of min/max sessions constraints on a single agent's U_i [0,1], 
    # we prioritize the optimization result shaped by the local BASE_DEMAND.
    
    # Ensure minimum usage based on the session requirement (if interpreted as minimum volume 0.1 per session)
    for t in range(SLOTS_PER_DAY):
        if min_sessions[t] > 0 and final_usage[t] < 0.1:
            final_usage[t] = 0.1
            
    return [min(1.0, u) for u in final_usage]


def main():
    # 1. Load Data
    # Assuming policy.py is run from the agent's directory
    scenario_data = load_scenario_data()
    
    # Extract relevant dynamic data (Global/Forecast)
    global_carbon = scenario_data['carbon_intensity']
    global_price = scenario_data['price']
    global_baseline = scenario_data['baseline_load']
    
    # Extract Neighbor and Agent specific data
    neighbor_info = get_neighbor_info(scenario_data)
    
    # Prepare Spatial Carbon data for each day, indexed by day (0 to 6)
    daily_spatial_carbon = []
    for i in range(1, DAYS_IN_WEEK + 1):
        day_key = f"Day {i} (Day {i} — "
        day_data = [v for k, v in scenario_data['days'].items() if k.startswith(day_key)][0]
        
        # Get the spatial carbon data relevant to this agent's location (LOCATION = 3)
        sc_data_dict = day_data['Spatial carbon']
        agent_sc = parse_spatial_carbon(sc_data_dict, LOCATION)
        daily_spatial_carbon.append(agent_sc)
        
    # 2. Generate Policy
    policy_output: List[List[float]] = []
    
    for day_index in range(DAYS_IN_WEEK):
        agent_usage = generate_usage_vector(
            day_index, 
            scenario_data, 
            neighbor_info, 
            daily_spatial_carbon[day_index]
        )
        policy_output.append(agent_usage)
        
    # 3. Save Output
    output_file = "global_policy_output.json"
    with open(output_file, 'w') as f:
        json.dump(policy_output, f, indent=4)

if __name__ == "__main__":
    main()