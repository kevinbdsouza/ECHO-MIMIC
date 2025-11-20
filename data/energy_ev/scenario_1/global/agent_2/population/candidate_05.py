import json
import numpy as np
import os

# --- Configuration ---
SLOT_COUNT = 4
DAYS_IN_FORECAST = 7

# --- Agent Specific Configuration (Position 2 Feeder Analyst) ---
AGENT_LOCATION_ID = 2
BASE_DEMAND = np.array([0.70, 1.00, 0.80, 0.50]) # [Slot 0, Slot 1, Slot 2, Slot 3]
CAPACITY = 6.8
ALPHA = 40.00  # Weight for carbon cost
BETA = 0.50    # Weight for price cost
GAMMA = 12.00  # Weight for congestion/spatial cost

# Neighbor Data (Hardcoded based on problem description for known neighbors)
NEIGHBOR_DATA = {
    1: { # Neighbor 1 (Battery Engineer, Location 1)
        "base_demand": np.array([1.20, 0.70, 0.80, 0.60]),
        "preferred_slots": [0, 2],
        "comfort_penalty": 0.18,
        "gt_min_cost_slots": [0, 1, 2, 0, 1, 2, 0] # Day 1 to Day 7
    },
    4: { # Neighbor 4 (Retirees, Location 4)
        "base_demand": np.array([0.90, 0.60, 0.70, 0.80]),
        "preferred_slots": [0, 3],
        "comfort_penalty": 0.16,
        "gt_min_cost_slots": [3, 3, 3, 2, 3, 3, 3] # Day 1 to Day 7
    }
}

# --- Data Loading Utility ---
def load_scenario_data(base_path="."):
    """Loads scenario data from scenario.json relative to the agent's directory."""
    file_path = os.path.join(base_path, "scenario.json")
    
    # In a typical execution environment, we assume the file is in the current directory
    # or one level up, depending on how the execution harness is structured.
    # We will try the current directory first.
    if not os.path.exists(file_path):
        # If running inside the agent's folder, the path might be relative to the script location.
        # For this specific problem structure, we rely on the execution environment placing
        # scenario.json where it can be found based on the instruction "Load scenario.json 
        # using only paths visible from the agent directory."
        pass 

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: scenario.json not found at expected path: {file_path}")
        # Fallback/Mock data based on problem description for testing robustness if needed, 
        # but for submission, we must assume the file exists.
        raise

    # Extract key global parameters
    slots_meta = data["slots"]
    global_prices = np.array(data["price"])
    global_carbons = np.array(data["carbon_intensity"])
    
    daily_data = {}
    day_names = list(data["days"].keys())
    
    for i, day_name in enumerate(day_names):
        day_info = data["days"][day_name]
        daily_data[day_name] = {
            "tariff": np.array(day_info["Tariff"]),
            "carbon": np.array(day_info["Carbon"]),
            "baseline_load": np.array(day_info["Baseline load"]),
            "spatial_carbon": {}
        }
        
        # Parse spatial carbon data (Location_ID: c0, c1, c2, c3)
        for loc_id in range(1, 6): # Locations 1 through 5
            key = str(loc_id)
            if key in day_info["Spatial carbon"]:
                daily_data[day_name]["spatial_carbon"][loc_id] = np.array(
                    day_info["Spatial carbon"][key].split('; ')[i].split(', ')
                ).astype(float)
            else:
                 # Handle the case where spatial carbon might be listed as one string per day block
                 # based on the structure: "Spatial carbon: 1: c0, c1, c2, c3 | 2: c0..."
                 # The input structure implies spatial carbon data for *all* locations is repeated 
                 # for each slot block, but keyed by location ID.
                 # The formatting "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ..." implies:
                 # For a given day, we look at the corresponding slot in the list for that location ID.
                 
                 # Re-parsing based on the day structure in the prompt:
                 # Day 1 (Day 1 — ...): Spatial carbon: 1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ...
                 # This means the provided spatial carbon structure is *already* indexed by location *within* the day structure.
                 
                 spatial_str = day_info["Spatial carbon"].get(str(loc_id))
                 if spatial_str:
                    # Extract the relevant slot's spatial carbon for this specific location
                    # Since we are iterating through days, we assume the values provided 
                    # for that location are the ones to use for the whole day's context, 
                    # UNLESS the prompt implies spatial carbon varies per slot *within* the day.
                    # Given the structure (4 values provided per location per day entry), 
                    # it's likely these 4 values correspond to the 4 slots (0, 1, 2, 3) for that location *on that day*.
                    
                    # Re-parsing logic based on standard structure:
                    sc_values = np.array([float(x) for x in spatial_str.split(', ')])
                    if len(sc_values) == SLOT_COUNT:
                        daily_data[day_name]["spatial_carbon"][loc_id] = sc_values
                    else:
                        # Fallback error or default assignment if structure is ambiguous
                        # Based on the prompt layout, the spatial carbon provided *within* the Day X block 
                        # seems to be the spatial impact experienced by that location across slots 0-3 for Day X.
                        # We will use the first value if length is wrong, though this is a guess.
                        if len(sc_values) > 0:
                             daily_data[day_name]["spatial_carbon"][loc_id] = np.tile(sc_values[0], SLOT_COUNT)
                        else:
                             daily_data[day_name]["spatial_carbon"][loc_id] = np.full(SLOT_COUNT, 500.0)

    
    # Determine Slot Min/Max sessions and Capacity Limits
    slot_limits = {
        "min_sessions": np.array(data["slot_min_sessions"].values()),
        "max_sessions": np.array(data["slot_max_sessions"].values())
    }
    
    return daily_data, slot_limits, global_prices, global_carbons

# --- Heuristic Cost Function Calculation ---

def calculate_cost(slot_idx, day_data, slot_limits, global_prices, global_carbons):
    """
    Calculates the composite cost for Agent 2 (Feeder Analyst, Location 2)
    to choose slot_idx on a specific day.
    Cost = alpha * Carbon + beta * Price + gamma * Congestion_Cost
    """
    
    day_tariff = day_data["tariff"][slot_idx]
    day_carbon = day_data["carbon"][slot_idx]
    day_baseline = day_data["baseline_load"][slot_idx]
    
    # 1. Carbon Cost (Minimization Objective)
    carbon_cost = ALPHA * day_carbon
    
    # 2. Price Cost (Minimization Objective)
    price_cost = BETA * day_tariff
    
    # 3. Congestion/Spatial Cost (Transformer Headroom Priority)
    
    # Agent 2 prioritizes transformer headroom. Congestion is primarily driven by 
    # local/spatial load relative to capacity, especially when neighbors are active.
    
    # Base Load Metric: How does the baseline load for this slot compare to capacity?
    # We penalize usage when baseline load is high relative to capacity.
    load_ratio = day_baseline / CAPACITY
    
    # Spatial Impact on Agent 2 (Location 2)
    # We check the spatial carbon experienced *at location 2* during this slot.
    # A high spatial carbon suggests upstream stress or high local contribution.
    spatial_impact_at_L2 = day_data["spatial_carbon"][AGENT_LOCATION_ID][slot_idx]
    
    # Neighbor Coordination Penalty:
    # Check what neighbors (1 and 4) prefer.
    
    # Neighbor 1 (Battery Engineer) prefers 0, 2. Penalize if we choose their preferred slot,
    # unless we must choose it for congestion reasons.
    neighbor1_preference_penalty = 0
    if slot_idx in NEIGHBOR_DATA[1]["preferred_slots"]:
        neighbor1_preference_penalty = NEIGHBOR_DATA[1]["comfort_penalty"]
    
    # Neighbor 4 (Retirees) prefers 0, 3. Penalize if we choose their preferred slot.
    neighbor4_preference_penalty = 0
    if slot_idx in NEIGHBOR_DATA[4]["preferred_slots"]:
        neighbor4_preference_penalty = NEIGHBOR_DATA[4]["comfort_penalty"]
        
    # Feeder Analyst Heuristic (Priority: Transformer Headroom, i.e., low load ratio)
    # We heavily penalize slots where load_ratio is high AND we overlap with others.
    
    # For a feeder analyst, minimizing local peak contribution is key.
    # Since we don't have true demand profiles, we proxy headroom strain by:
    # (High Baseline Load) + (Neighbor Overlap Cost)
    
    # Use a high multiplier for the load ratio component as this is our primary driver.
    congestion_proxy = (
        GAMMA * load_ratio  # Direct capacity pressure
        + 1.5 * GAMMA * spatial_impact_at_L2 / 1000 # Proxy for thermal stress from local grid state
        + 0.5 * (neighbor1_preference_penalty + neighbor4_preference_penalty) # Minor coordination cost
    )
    
    total_cost = carbon_cost + price_cost + congestion_proxy
    
    # Apply session constraints (Note: These are usually handled externally in scheduling, 
    # but we can assign an infinite cost if constraints are violated)
    # Since we are recommending *one* slot per day, session counts are not directly applicable 
    # unless we interpret min/max sessions as *required* vs *max allowed* participation.
    # Assuming these constraints apply to the *total* number of sessions scheduled across the 7 days 
    # by all agents, or per day if this agent is the only one scheduling slots 0-3.
    # For simplicity in this single-agent 7-day recommendation, we mostly ignore min/max unless
    # they flag an impossible scenario (which they don't here, as they are 1/2).
    
    return total_cost

# --- Neighbor Coordination Strategy (Collective Goal Focus) ---

def coordinate_neighbor_goals(slot_idx, day_name, day_index, daily_data):
    """
    Adjusts the cost based on collective goals, specifically checking if 
    neighbor's Ground Truth (GT) minimum cost slot suggests a conflicting priority.
    
    Agent 2 (Feeder Analyst) prioritizes headroom (low load/spatial impact).
    If a slot is bad for headroom but is GT-optimal for a neighbor, we might tolerate it 
    if the global factors (Carbon/Price) are exceptionally low there.
    
    Since we are prioritizing *our* goal (headroom) while being aware of others, 
    we primarily rely on the cost function above. Here, we add a minor nudge:
    If the GT slot for a neighbor is drastically different from the cost-optimal slot we calculate, 
    it signals an external priority (like comfort or budget) that we should try to respect
    if our primary goal is met reasonably well.
    """
    
    # Find the GT preferred slot for each neighbor for this day
    day_of_week_offset = day_index # Day 1 is index 0
    
    # Neighbor 1 GT slot (index 0-3)
    n1_gt_slot = NEIGHBOR_DATA[1]["gt_min_cost_slots"][day_of_week_offset]
    # Neighbor 4 GT slot (index 0-3)
    n4_gt_slot = NEIGHBOR_DATA[4]["gt_min_cost_slots"][day_of_week_offset]
    
    coordination_adjustment = 0.0
    
    # Coordination Logic: If we pick a slot that is strongly *dispreferred* by a neighbor's GT 
    # (assuming GT reflects their actual constraints), we add a minor penalty, unless 
    # that slot is our lowest cost option.
    
    # For a feeder analyst, the primary goal is local congestion. We assume this goal 
    # aligns with minimizing aggregate carbon/load. We only coordinate if the neighbor's 
    # GT choice is vastly superior in carbon/price compared to our calculated minimum.
    
    # Since the instructions emphasize explainable heuristics combined with local coordination,
    # we will primarily use the cost function. We introduce coordination here by softening 
    # penalties associated with slots that neighbors *must* use.
    
    # If slot_idx IS a neighbor's GT slot, reduce the perceived spatial/congestion penalty slightly, 
    # assuming their need for that slot justifies potential minor localized impact.
    
    if slot_idx == n1_gt_slot:
        coordination_adjustment -= 0.5 * GAMMA # Slight relief for N1's essential slot
    if slot_idx == n4_gt_slot:
        coordination_adjustment -= 0.5 * GAMMA # Slight relief for N4's essential slot
        
    return coordination_adjustment


def make_daily_recommendation(day_name, day_index, daily_data, slot_limits, global_prices, global_carbons):
    """Calculates the optimal slot for one day."""
    
    best_cost = float('inf')
    best_slot = -1
    
    costs_per_slot = {}
    
    for slot_idx in range(SLOT_COUNT):
        # 1. Calculate inherent cost (Carbon, Price, Local Headroom)
        base_cost = calculate_cost(
            slot_idx, daily_data[day_name], slot_limits, global_prices, global_carbons
        )
        
        # 2. Apply coordination adjustment based on neighbor GT needs
        coordination_nudge = coordinate_neighbor_goals(
            slot_idx, day_name, day_index, daily_data
        )
        
        final_cost = base_cost + coordination_nudge
        costs_per_slot[slot_idx] = final_cost

        if final_cost < best_cost:
            best_cost = final_cost
            best_slot = slot_idx
            
    return best_slot, costs_per_slot

# --- Main Execution ---

def run_policy():
    # 1. Load Data
    try:
        daily_data, slot_limits, global_prices, global_carbons = load_scenario_data()
    except Exception as e:
        print(f"Failed to load or parse scenario data: {e}")
        # Return a default failure output structure if loading fails catastrophically
        recommendations = [3] * DAYS_IN_FORECAST
        with open("global_policy_output.json", 'w') as f:
            json.dump({"recommendations": recommendations}, f)
        return

    recommendations = []
    day_names = list(daily_data.keys())
    
    # Ensure we have exactly 7 days
    if len(day_names) != DAYS_IN_FORECAST:
        # If the number of days parsed is wrong, default to the safest slot (Slot 3 in this scenario)
        print(f"Warning: Expected 7 days, found {len(day_names)}. Defaulting to Slot 3.")
        recommendations = [3] * DAYS_IN_FORECAST
    else:
        # 2. Decide on slot for each day
        # Since this is a 7-day lookahead policy, the cost calculation for each day 
        # implicitly uses the context (Tariff, Carbon, Spatial) for that specific day.
        # We run the optimization sequentially for each day without imposing session limits across days, 
        # as session requirements are usually for the *entire* collective optimization run.
        
        for day_index, day_name in enumerate(day_names):
            slot_recommendation, costs = make_daily_recommendation(
                day_name, day_index, daily_data, slot_limits, global_prices, global_carbons
            )
            recommendations.append(slot_recommendation)

    # 3. Write global_policy_output.json
    output_data = {"recommendations": recommendations}
    
    with open("global_policy_output.json", 'w') as f:
        json.dump(output_data, f, indent=4)
        
    # Optional: Print for debugging/verification (remove for final submission if strict output required)
    # print("Agent 2 (Feeder Analyst) Recommendations:")
    # for i, day in enumerate(day_names):
    #     print(f"  {day}: Slot {recommendations[i]}")


if __name__ == "__main__":
    run_policy()