import json
import numpy as np
from collections import defaultdict

# --- Configuration ---
# This agent is Position 1, a battery engineer balancing budget and solar backfeed.
AGENT_ID = 1
CAPACITY = 6.8
# Base demand for Agent 1 (Position 1)
BASE_DEMAND = np.array([1.20, 0.70, 0.80, 0.60])

# Coordination parameters (from scenario)
ALPHA = 40.00  # Price/Carbon sensitivity
BETA = 0.50    # Spatial Carbon sensitivity
GAMMA = 12.00  # Comfort/Baseline sensitivity

# Neighbor data (manually parsed from prompt)
NEIGHBOR_DATA = {
    2: {
        'location': 2,
        'base_demand': np.array([0.70, 1.00, 0.80, 0.50]),
        'preferred_slots': [1, 2],
        'comfort_penalty': 0.14,
        'ground_truth_min_cost': [1, 2, 0, 1, 2, 0, 1] # Day 1 to Day 7 index
    },
    3: {
        'location': 3,
        'base_demand': np.array([0.60, 0.80, 0.90, 0.70]),
        'preferred_slots': [1, 3],
        'comfort_penalty': 0.20,
        'ground_truth_min_cost': [2, 0, 1, 3, 0, 1, 2] # Day 1 to Day 7 index
    }
}

# --- Helper Functions ---

def load_scenario_data(filename="scenario.json"):
    """Loads scenario data from the specified JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def parse_spatial_carbon(sc_str, num_neighbors=5):
    """Parses the spatial carbon string into a structured dictionary."""
    # Format: 1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | ...
    spatial_data = {}
    parts = sc_str.split(' | ')
    for part in parts:
        try:
            neighbor_id_str, values_str = part.split(': ')
            neighbor_id = int(neighbor_id_str)
            values = [int(v.strip()) for v in values_str.split(',')]
            spatial_data[neighbor_id] = np.array(values)
        except ValueError:
            # Handle cases where parsing fails unexpectedly, though unlikely with this structure
            continue
    return spatial_data

def calculate_load(agent_demand, slot_occupancy):
    """Calculates the total load for the agent given its demand profile and slot occupancy."""
    # Assuming slot_occupancy is 1 if the slot is chosen, 0 otherwise.
    # The load incurred is the agent's base demand plus the load added by the session (if selected).
    # In this context (choosing one slot), we calculate the total energy drawn in that slot.
    # Since we are choosing ONE slot for the day, we use the base demand + added load for that slot.
    
    # Since the problem defines baseline_load as the expected load *without* the agent's intervention,
    # and we are deciding how much to charge, a common approach is to calculate the net impact.
    # However, for coordination, we often look at the total load *including* the agent's charge.
    
    # Given the setup, we assume the agent draws its *full* required energy (BASE_DEMAND[i] + session_energy)
    # in the chosen slot i. Since we don't know the *required energy* per slot, we must assume
    # that the total energy drawn in the chosen slot `i` is related to BASE_DEMAND[i].
    
    # Let's simplify based on common practice: the load generated if we choose slot i is
    # the base demand + the energy required to satisfy the load requirement in that slot.
    # Without a specific required energy, we must use the agent's demand vector as the load profile *if* it charges.
    
    if slot_occupancy == 1:
        return agent_demand
    return np.zeros_like(agent_demand) # Should not happen if only one slot is chosen

# --- Scoring Function ---

def score_slot(day_data, slot_index, current_day_name, neighbor_spatial_data):
    """
    Calculates the heuristic score for choosing a specific slot on a given day.
    Score minimizes (ALPHA * Price + Carbon) + (GAMMA * Local Congestion) + (BETA * Neighbor Congestion)
    """
    S = 4  # Number of slots
    
    # 1. Local Cost (Price and Carbon Intensity) - Weighted by ALPHA
    # We use the worst-case (highest) expected value ±20% noise. We use the forecast value.
    price = day_data['Tariff'][slot_index]
    carbon = day_data['Carbon'][slot_index]
    
    local_cost = ALPHA * price + carbon

    # 2. Local Congestion (Baseline Load Impact) - Weighted by GAMMA
    # The agent assumes it will charge its full load (BASE_DEMAND) in the selected slot.
    # Congestion is proportional to how much the baseline load in that slot exceeds capacity relative to the load added.
    
    # Since capacity (6.8) is the feeder limit, and baseline load is provided,
    # we evaluate the total load: Baseline + Agent Load.
    
    baseline_load = day_data['Baseline load'][slot_index]
    agent_load_in_slot = BASE_DEMAND[slot_index] # Energy drawn by Agent 1 in this slot
    
    total_load = baseline_load + agent_load_in_slot
    
    # Congestion Penalty: Penalize high total load, normalized by capacity.
    # We add a comfort penalty (gamma * load) term.
    local_congestion_penalty = GAMMA * (total_load / CAPACITY)

    # 3. Neighbor Coordination (Spatial Carbon) - Weighted by BETA
    neighbor_carbon_penalty = 0.0
    
    # Agent 1 is Location 1. We look at spatial carbon data provided for other locations.
    # We need the spatial carbon data *for the chosen slot*.
    
    # Spatial carbon data is provided for neighbors (locations 2-5, etc., depending on scenario structure).
    # We must check which neighbors provided data relative to our location (1).
    
    # The spatial carbon array provided usually represents the congestion cost AT THAT LOCATION
    # IF a neighbor charges in that slot. Since we are Location 1, we should look at
    # the influence of *our* charge on *neighbors'* spatial carbon metrics if we knew how they were calculated.
    
    # Based on standard Echo interpretation: BETA relates to the penalty imposed on the *network* by our action.
    # Since we are location 1, we check the spatial carbon values associated with our chosen slot
    # across all neighbors' reported spatial carbon arrays.
    
    spatial_carbon_sum = 0.0
    neighbor_count = 0
    
    # Aggregate spatial carbon reported by neighbors for this slot
    for neighbor_id, spatial_data in neighbor_spatial_data.items():
        if slot_index in spatial_data:
            # spatial_data[slot_index] is the carbon impact score at the neighbor's location for this slot.
            spatial_carbon_sum += spatial_data[slot_index]
            neighbor_count += 1

    if neighbor_count > 0:
        # Penalize based on the average negative impact our choice imposes on neighbors' spatial carbon metrics.
        neighbor_carbon_penalty = BETA * (spatial_carbon_sum / neighbor_count)
    
    # 4. Comfort/Preference Penalty (Implicit - not explicitly parameterized by ALPHA/BETA/GAMMA for Agent 1,
    # but we can add a small penalty if the slot is outside neighbor preferences for coordination)
    # As a battery engineer, minimizing cost/carbon is key, but we look at neighbor preferences as soft targets.
    
    coordination_preference_penalty = 0.0
    
    # Check against neighbors' preferred slots (using inverse preference: lower score for preferred slots)
    # We don't have an explicit comfort term for Agent 1, but we can use neighbors' comfort penalties to modulate.
    # For simplicity, we only implement the explicit cost parameters ALPHA, BETA, GAMMA.
    
    total_score = local_cost + local_congestion_penalty + neighbor_carbon_penalty
    
    return total_score

def get_day_data(scenario_data, day_index):
    """Extracts relevant dynamic data for a specific day."""
    day_key = f"Day {day_index + 1}"
    
    # Extract the specific day's environmental data
    day_data = scenario_data['days'][day_key]
    
    # Combine forecast (header) and day-specific data, prioritizing day-specific data
    # Slot mapping: 0, 1, 2, 3 correspond to the 4 evening slots.
    
    data = {
        'Tariff': np.array(day_data['Tariff']),
        'Carbon': np.array(day_data['Carbon']),
        'Baseline load': np.array(day_data['Baseline load']),
    }
    
    # Parse spatial carbon strings into dictionaries of arrays
    forecast_spatial = parse_spatial_carbon(scenario_data['spatial_carbon'])
    day_spatial = {}
    
    for nid_str, values_str in [p.split(':') for p in day_data['Spatial carbon'].split(';')]:
        neighbor_id = int(nid_str.strip())
        values = [int(v.strip()) for v in values_str.split(',')]
        day_spatial[neighbor_id] = np.array(values)
        
    # Spatial coordination for Agent 1 (Location 1) requires pooling influence data.
    # We use the spatial data provided by neighbors (which includes spatial data for all locations 1-5)
    # We need the spatial influence data reported *by neighbors* for this day.
    
    return data, day_spatial


# --- Main Policy Logic ---

def generate_policy():
    
    # 1. Load Scenario
    try:
        scenario = load_scenario_data("scenario.json")
    except FileNotFoundError:
        # If running outside the expected directory structure, try relative path
        try:
            scenario = load_scenario_data("./scenario.json")
        except FileNotFoundError:
            print("Error: scenario.json not found.")
            return

    NUM_DAYS = 7
    NUM_SLOTS = len(scenario['slots'])
    
    # Determine agent's location from spatial carbon structure (assuming location 1 based on role)
    # We assume Agent 1 is Location 1 for spatial coordination calculations.
    AGENT_LOCATION = AGENT_ID 
    
    recommendations = {}
    
    # Iterate over the 7 days
    for day_idx in range(NUM_DAYS):
        day_name = f"Day {day_idx + 1}"
        
        day_data, day_spatial_data = get_day_data(scenario, day_idx)
        
        best_score = float('inf')
        best_slot = -1
        
        slot_scores = {}

        # 2. Evaluate each slot for the current day
        for slot_idx in range(NUM_SLOTS):
            
            # Check constraints (Min/Max sessions are typically constraints for aggregate charging,
            # but for a single agent choosing one slot, we assume constraints are met if we pick one.)
            # We skip explicit checking of slot_min/max sessions since the agent only chooses one slot per day.
            
            score = score_slot(
                day_data, 
                slot_idx, 
                day_name, 
                day_spatial_data
            )
            
            slot_scores[slot_idx] = score
            
            if score < best_score:
                best_score = score
                best_slot = slot_idx
        
        recommendations[day_name] = best_slot

    # 3. Format output for global_policy_output.json
    # Ensure the output is a list of 7 slot indices corresponding to Day 1 to Day 7
    output_slots = [recommendations[f"Day {i+1}"] for i in range(NUM_DAYS)]
    
    # 4. Save results
    output_filename = "global_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(output_slots, f, indent=4)
        
    # The script should only return the Python source, so the execution ends here.

# --- Execution ---
generate_policy()
# End of policy.py content
return # Placeholder for script boundary if this were a function, but here it just executes.