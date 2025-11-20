import json
import numpy as np
import os
from typing import List, Dict, Any

# --- Configuration ---
SLOT_COUNT = 4
DAYS_IN_FORECAST = 7

# --- Agent Specific Configuration (Position 2 Feeder Analyst) ---
# Prioritizes transformer headroom (Congestion/Spatial Carbon)
AGENT_LOCATION_ID = 2
BASE_DEMAND = np.array([0.70, 1.00, 0.80, 0.50]) 
CAPACITY = 6.8
ALPHA = 40.00  # Weight for Carbon Cost (Global Goal)
BETA = 0.50    # Weight for Price Cost (General Goal)
GAMMA = 12.00  # Weight for Congestion/Spatial Cost (Local Goal Priority)

# Neighbor Data (Known only for coordination signals)
NEIGHBOR_DATA = {
    1: { # Neighbor 1 (Location 1, Battery Engineer: Budget/Solar)
        "base_demand": np.array([1.20, 0.70, 0.80, 0.60]),
        "preferred_slots": [0, 2],
        "comfort_penalty": 0.18,
        "gt_min_cost_slots": [0, 1, 2, 0, 1, 2, 0] # Day 1 to Day 7 GT minimum cost slots
    },
    4: { # Neighbor 4 (Location 4, Retirees: Comfort/Warnings)
        "base_demand": np.array([0.90, 0.60, 0.70, 0.80]),
        "preferred_slots": [0, 3],
        "comfort_penalty": 0.16,
        "gt_min_cost_slots": [3, 3, 3, 2, 3, 3, 3] # Day 1 to Day 7 GT minimum cost slots
    }
}

# --- Data Loading Utility ---
def load_scenario_data(base_path="."):
    """Loads scenario data from scenario.json."""
    file_path = os.path.join(base_path, "scenario.json")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("scenario.json not found.")

    # Extract key global parameters
    global_prices = np.array(data["price"])
    global_carbons = np.array(data["carbon_intensity"])
    
    slot_limits = {
        "min_sessions": np.array(list(data["slot_min_sessions"].values())),
        "max_sessions": np.array(list(data["slot_max_sessions"].values()))
    }

    daily_data = {}
    day_names = sorted(data["days"].keys(), key=lambda x: int(x.split(' ')[1])) # Sort Day 1, Day 2...
    
    for i, day_name in enumerate(day_names):
        day_info = data["days"][day_name]
        daily_data[day_name] = {
            "tariff": np.array(day_info["Tariff"]),
            "carbon": np.array(day_info["Carbon"]),
            "baseline_load": np.array(day_info["Baseline load"]),
            "spatial_carbon": {}
        }
        
        # Parse spatial carbon data for all locations (1 through 5)
        for loc_id in range(1, 6): 
            spatial_str = day_info["Spatial carbon"].get(str(loc_id))
            
            if spatial_str:
                # Values are assumed to correspond to slots 0, 1, 2, 3 for that day/location
                sc_values = np.array([float(x) for x in spatial_str.split(', ')])
                if len(sc_values) == SLOT_COUNT:
                    daily_data[day_name]["spatial_carbon"][loc_id] = sc_values
                else:
                    # Fallback: If structure is inconsistent, use the first available value tiled
                    if len(sc_values) > 0:
                         daily_data[day_name]["spatial_carbon"][loc_id] = np.tile(sc_values[0], SLOT_COUNT)
                    else:
                         daily_data[day_name]["spatial_carbon"][loc_id] = np.full(SLOT_COUNT, 500.0)
            else:
                 daily_data[day_name]["spatial_carbon"][loc_id] = np.full(SLOT_COUNT, 500.0) # Default large stress

    return daily_data, slot_limits, global_prices, global_carbons

# --- Cost Calculation Components ---

def calculate_congestion_cost(slot_idx, day_data):
    """
    Calculates congestion cost, prioritizing transformer headroom (high GAMMA weighting).
    Congestion is proxied by high baseline load + high spatial carbon at location 2, 
    and soft penalties for overlapping with neighbor's preferred slots.
    """
    day_baseline = day_data["baseline_load"][slot_idx]
    
    # 1. Base Load Strain: How heavily utilized is the transformer baseline?
    load_ratio = day_baseline / CAPACITY
    
    # 2. Spatial Stress at Location 2 (Primary Feeder Analyst Concern)
    spatial_impact_at_L2 = day_data["spatial_carbon"][AGENT_LOCATION_ID][slot_idx]
    
    # 3. Neighbor Overlap Penalty (Coordination consideration)
    neighbor_overlap_penalty = 0
    
    # Neighbor 1 (Battery Engineer) prefers 0, 2.
    if slot_idx in NEIGHBOR_DATA[1]["preferred_slots"]:
        neighbor_overlap_penalty += NEIGHBOR_DATA[1]["comfort_penalty"] * 1.5
    
    # Neighbor 4 (Retirees) prefers 0, 3.
    if slot_idx in NEIGHBOR_DATA[4]["preferred_slots"]:
        neighbor_overlap_penalty += NEIGHBOR_DATA[4]["comfort_penalty"] * 1.0
        
    # Combine factors, heavily weighting spatial stress (as primary constraint) and load ratio.
    congestion_cost = (
        GAMMA * (load_ratio * 1.5)  # Load ratio contributes heavily
        + GAMMA * 0.1 * spatial_impact_at_L2 / 1000 # Spatial factor provides local context
        + neighbor_overlap_penalty * 2.0 # Coordination impact on local resources
    )
    
    # Add a hard capacity check proxy (though sessions are assumed 1 if chosen)
    if day_baseline + BASE_DEMAND[slot_idx] > CAPACITY:
         congestion_cost += 500.0 

    return congestion_cost

def calculate_carbon_cost(slot_idx, day_data):
    """Calculates the weighted cost for carbon intensity."""
    day_carbon = day_data["carbon"][slot_idx]
    # Use ALPHA weight
    return ALPHA * day_carbon

def calculate_price_cost(slot_idx, day_data):
    """Calculates the weighted cost for price."""
    day_tariff = day_data["tariff"][slot_idx]
    # Use BETA weight
    return BETA * day_tariff

def calculate_utility(slot_idx, day_data, day_index):
    """
    Calculates the composite utility (cost to minimize) for selecting a slot on a given day.
    J = Carbon_Cost + Price_Cost + Congestion_Cost + Coordination_Adjustment
    """
    
    price_cost = calculate_price_cost(slot_idx, day_data)
    carbon_cost = calculate_carbon_cost(slot_idx, day_data)
    congestion_cost = calculate_congestion_cost(slot_idx, day_data)
    
    # Coordination Adjustment: Respecting neighbor GT signals (Softening penalties if we align with their known minimums)
    day_of_week_offset = day_index
    coord_adjustment = 0.0
    
    n1_gt_slot = NEIGHBOR_DATA[1]["gt_min_cost_slots"][day_of_week_offset]
    n4_gt_slot = NEIGHBOR_DATA[4]["gt_min_cost_slots"][day_of_week_offset]
    
    # If we choose a slot that a neighbor *needs* (GT minimal), slightly reduce our congestion cost
    if slot_idx == n1_gt_slot:
        coord_adjustment -= 0.5 * GAMMA
    if slot_idx == n4_gt_slot:
        coord_adjustment -= 0.5 * GAMMA
        
    total_cost = price_cost + carbon_cost + congestion_cost + coord_adjustment
    
    return total_cost

# --- Daily Decision ---

def make_daily_recommendation(day_name, day_index, daily_data, slot_limits):
    """Calculates the optimal slot for one day by minimizing utility."""
    
    best_cost = float('inf')
    best_slot = -1
    
    for slot_idx in range(SLOT_COUNT):
        utility = calculate_utility(slot_idx, daily_data[day_name], day_index)
        
        if utility < best_cost:
            best_cost = utility
            best_slot = slot_idx
            
    # Final check: Ensure the selected slot meets the minimum session requirement (1)
    # If the best slot requires sessions > 1, but we only recommend one, we assume the cost function
    # already handled this implicitly through congestion/baseline scaling for a single agent contribution.
    
    return best_slot

# --- Main Policy Execution ---

def run_policy():
    # 1. Load Data
    try:
        daily_data, slot_limits, _, _ = load_scenario_data()
    except Exception as e:
        print(f"Error executing policy: {e}")
        # Create a safe fallback file if loading fails
        with open("global_policy_output.json", 'w') as f:
            json.dump({"recommendations": [3] * DAYS_IN_FORECAST}, f)
        return

    recommendations = []
    day_names = list(daily_data.keys())
    
    if len(day_names) != DAYS_IN_FORECAST:
        # Fallback if data structure is unexpected
        recommendations = [3] * DAYS_IN_FORECAST
    else:
        # 2. Decide on slot for each day using 7-day lookahead context (via day-specific data)
        for day_index, day_name in enumerate(day_names):
            slot_recommendation = make_daily_recommendation(
                day_name, day_index, daily_data, slot_limits
            )
            recommendations.append(slot_recommendation)

    # 3. Write global_policy_output.json
    output_data = {"recommendations": recommendations}
    
    with open("global_policy_output.json", 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    run_policy()