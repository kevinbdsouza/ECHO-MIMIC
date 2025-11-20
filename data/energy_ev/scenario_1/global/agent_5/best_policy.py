import json
import os
from itertools import product

# --- Configuration ---
SLOT_COUNT = 4
DAY_COUNT = 7
CAPACITY = 6.8

# --- Agent 5 Specifics ---
AGENT_ID = 5
BASE_DEMAND = [0.50, 0.70, 0.60, 0.90]
# Position 5 tenant commuting late, likely wants to charge later if possible, but comfort is key.
# Given the high base demand in slot 3 (0.90), leaning slightly away from the latest slot if costs are high.
# Preferences derived from persona: Moderate comfort sensitivity, higher demand flexibility than retirees, less budget-driven than engineer.
# Let's prioritize slots 2 and 3 (later times) slightly, but heavily weight carbon/price.

# --- Utility Functions ---

def load_scenario_data(base_path="./"):
    """Loads scenario data from JSON files relative to the execution directory."""
    try:
        # Assuming scenario.json contains all fixed/common data
        with open(os.path.join(base_path, "scenario.json"), 'r') as f:
            scenario = json.load(f)
        
        # Reconstruct the full 7-day structure based on the scenario file content if needed,
        # but for simplicity in this context, we often rely on the structure provided in the prompt.
        # Since the prompt gives a structured view, we parse it directly into a usable structure.
        
        # Extract fixed data
        fixed_data = {
            "slots": scenario["slots"],
            "price": scenario["price"],
            "carbon_intensity": scenario["carbon_intensity"],
            "capacity": scenario["capacity"],
            "baseline_load": scenario["baseline_load"],
            "slot_min_sessions": scenario["slot_min_sessions"],
            "slot_max_sessions": scenario["slot_max_sessions"],
            "spatial_carbon_template": scenario["spatial_carbon"], # This needs parsing per day
            "alpha": scenario["alpha"],
            "beta": scenario["beta"],
            "gamma": scenario["gamma"],
        }

        # Parse daily data
        daily_data = {}
        day_names = list(scenario["days"].keys())
        for day_name in day_names:
            day_info = scenario["days"][day_name]
            
            # Parse spatial carbon strings into lists of floats/ints
            spatial_carbon_map = {}
            for loc_key, sc_string in day_info["Spatial carbon"].items():
                try:
                    # Spatial carbon is formatted like "1: 330, 520, 560, 610"
                    loc_id = int(loc_key.split(':')[0].strip())
                    sc_values = [float(x) for x in sc_string.split('; ')[0].split(', ')]
                    spatial_carbon_map[loc_id] = sc_values
                except Exception:
                    # Handle cases where the prompt format might be slightly different in the JSON structure
                    # Fallback to assuming the provided string is space/comma separated values for the 4 slots
                    try:
                        sc_values = [float(x) for x in sc_string.replace(';', '').split(', ')]
                        spatial_carbon_map[loc_id] = sc_values
                    except:
                         # If direct parsing fails, use global for this location if available
                        spatial_carbon_map[loc_id] = scenario["spatial_carbon"][str(loc_id)]


            daily_data[day_name] = {
                "Tariff": [float(t) for t in day_info["Tariff"].split(', ')],
                "Carbon": [float(c) for c in day_info["Carbon"].split(', ')],
                "Baseline load": [float(b) for b in day_info["Baseline load"].split(', ')],
                "Spatial carbon": spatial_carbon_map,
            }

        return fixed_data, daily_data, day_names

    except FileNotFoundError:
        print(f"Error: scenario.json not found in {os.getcwd()}. Ensure the file structure is correct.")
        # Return dummy structure if loading fails to allow script execution for testing structure
        return None, None, None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None, None

def calculate_agent_load(slot_index, demand_profile):
    """Calculates the total energy demand for the agent in a specific slot."""
    # Since we are making a single decision (session count 1 or 0), 
    # we use base demand directly, assuming 1 session if chosen.
    return demand_profile[slot_index]

def calculate_neighbor_load_projection(day_data, day_name, neighbor_examples, current_slot_index):
    """
    Projects neighbor load for the current slot based on their ground truth (GT)
    historical choices, adjusted for the current day's context if possible.
    
    Since we are in Stage 3 (Collective), we must use neighbor examples to infer
    their likely behavior for *this* specific day, even though the examples
    only provide GT for past days.
    """
    
    neighbor_loads = {loc: 0.0 for loc in [1, 2, 3, 4]} # We coordinate with 1 and 4
    
    # Mapping neighbor ID to its data structure in the scenario
    N_MAP = {
        4: {"Base demand": [0.90, 0.60, 0.70, 0.80], "Comfort penalty": 0.16, "Preferred slots": [0, 3]},
        1: {"Base demand": [1.20, 0.70, 0.80, 0.60], "Comfort penalty": 0.18, "Preferred slots": [0, 2]}
    }
    
    # The GT provides the *actual* chosen slot for the past 7 days.
    # We assume the agent will follow its historical minimum-cost strategy unless
    # current conditions strongly deviate, which is hard to model perfectly here.
    # Simplification: Assume neighbors stick closely to their historical "min-cost" slot for this day of the week.
    
    # Get the GT slot for the current day (Day N, where N=1..7)
    day_index = int(day_name.split(' ')[1].strip(' (')) - 1 # 0 to 6

    for n_id, n_data in neighbor_examples.items():
        if str(n_id) not in n_data: continue
        
        # Extract GT slot for this day
        gt_slots = n_data[str(n_id)]["ground_truth_min_cost_slots"]
        
        if day_index < len(gt_slots):
            # Assume the neighbor chooses the historically best slot for this day index
            chosen_slot = gt_slots[day_index]
            
            # Get neighbor's base demand for that chosen slot
            neighbor_demand = N_MAP[n_id]["Base demand"][chosen_slot]
            neighbor_loads[n_id] = neighbor_demand
            
    # Sum the projected load from neighbors coordinating with us
    total_neighbor_load = sum(neighbor_loads.values())
    
    return total_neighbor_load

# --- Heuristic Policy Definition ---

def heuristic_score(slot_index, day_data, fixed_data, agent_demand, total_neighbor_load, day_name):
    """
    Calculates a composite score for a slot, prioritizing:
    1. Carbon intensity (lowest is best).
    2. Congestion (Total Load vs Capacity).
    3. Price (lowest is best, weighted lower than Carbon/Congestion).
    
    Weights based on global tuning parameters (alpha, beta, gamma):
    Cost = alpha * Carbon + beta * Congestion + gamma * Price
    """
    
    # 1. Carbon Intensity (Global Goal)
    carbon = day_data["Carbon"][slot_index]
    
    # 2. Congestion (Grid Health / Spatial awareness)
    # Total Load = Agent Load + Neighbor Load + Baseline Load
    # Spatial Carbon reflects local grid stress due to demand distribution. We use it as a congestion proxy.
    
    # Spatial carbon for Agent 5 (Location 5) on this day
    try:
        spatial_c5 = day_data["Spatial carbon"][AGENT_ID][slot_index]
    except KeyError:
        # Fallback to global carbon if location-specific data isn't parsed correctly
        spatial_c5 = day_data["Carbon"][slot_index]

    # Total load projection (highly simplified for coordination stage 3)
    baseline = day_data["Baseline load"][slot_index]
    # We approximate the total system load we are contributing to:
    # Current Agent Load + Projected Neighbor Load + Baseline Load + Current Agent Load (since agent load contributes to congestion)
    projected_load = agent_demand + total_neighbor_load + baseline
    
    # Normalize congestion: We want load below capacity, penalized heavily if above.
    # Beta term emphasizes avoiding exceeding capacity.
    if projected_load > CAPACITY:
        congestion_penalty = beta * (projected_load - CAPACITY) * 10.0 # High penalty for exceeding capacity
    else:
        # Mild penalty for simply being high, relative to baseline
        congestion_penalty = beta * (projected_load / CAPACITY)
        
    # 3. Price (Local Cost)
    price = day_data["Tariff"][slot_index]
    
    # 4. Comfort/Personal Preference (Implicit: Agent 5 favors later slots 2, 3, but this is weakly enforced)
    # Since we don't have a comfort score defined for Agent 5 in the prompt examples, we rely on environmentals.
    # We will apply a slight bias against slot 0 if conditions are otherwise good.
    comfort_adjustment = 0.0
    if slot_index == 0:
        comfort_adjustment = 5.0 # Small penalty for the earliest slot
        
    # --- Final Score Calculation (Minimize Score) ---
    # Components must be scaled appropriately based on typical ranges.
    # Carbon range: ~450 to 750
    # Price range: ~0.20 to 0.32
    # Congestion term (projection): Range ~5 to 15 (normalized) + penalty
    
    # Scaling factors based on alpha=40, beta=0.5, gamma=12:
    
    score = (
        alpha * carbon +                # Heavy weight on carbon (40x)
        beta * projected_load * 100 +   # Weighting load relative to magnitude (0.5x * Load * 100 -> Load scale 500-1500)
        gamma * price * 100 +           # Weighting price (12x * Price * 100 -> Price scale 2000-3200)
        comfort_adjustment              # Small comfort adjustment
    )
    
    # Add spatial carbon as a secondary congestion/local grid health metric
    score += 2.0 * spatial_c5 
    
    return score

def determine_recommendation(fixed_data, daily_data, day_names, neighbor_examples):
    """Determines the best slot for each of the 7 days."""
    
    recommendations = {}
    
    # Pre-calculate Agent 5's load for each slot (assuming session=1)
    agent_loads = [calculate_agent_load(i, BASE_DEMAND) for i in range(SLOT_COUNT)]

    # Iterate through each day
    for day_index, day_name in enumerate(day_names):
        day_data = daily_data[day_name]
        
        best_score = float('inf')
        best_slot = -1
        
        # --- Coordination Step: Estimate Neighbor Load ---
        # Since coordination is only *with* neighbors based on observed examples, 
        # we project what they *likely* did on this day based on their GT history.
        
        # Neighbor examples need to be mapped to the scenario structure
        processed_neighbors = {}
        for n_ex in neighbor_examples:
            # Assuming neighbor_examples is a list of dicts, extract relevant info
            
            # Neighbor 4
            if 'Neighbor 4' in n_ex:
                processed_neighbors[4] = {
                    "Base demand": n_ex['Neighbor 4'][0]['Base demand'], 
                    "Comfort penalty": n_ex['Neighbor 4'][0]['Comfort penalty'],
                    "Preferred slots": n_ex['Neighbor 4'][0]['Preferred slots'],
                    "ground_truth_min_cost_slots": n_ex['Neighbor 4'][0]['ground_truth min-cost slots by day']
                }
            # Neighbor 1
            elif 'Neighbor 1' in n_ex:
                processed_neighbors[1] = {
                    "Base demand": n_ex['Neighbor 1'][0]['Base demand'], 
                    "Comfort penalty": n_ex['Neighbor 1'][0]['Comfort penalty'],
                    "Preferred slots": n_ex['Neighbor 1'][0]['Preferred slots'],
                    "ground_truth_min_cost_slots": n_ex['Neighbor 1'][0]['ground_truth min-cost slots by day']
                }

        # This projection is crucial for coordination. We use the GT slots provided for the corresponding day index.
        total_neighbor_load = calculate_neighbor_load_projection(
            day_data, day_name, processed_neighbors, -1
        )
        
        # --- Slot Evaluation ---
        for slot in range(SLOT_COUNT):
            agent_load = agent_loads[slot]
            
            # Score calculation based on current day's data and projected congestion
            score = heuristic_score(
                slot, 
                day_data, 
                fixed_data, 
                agent_load, 
                total_neighbor_load,
                day_name
            )
            
            # Optional: Apply slight preference adjustment for Agent 5 (prefers later slots 2, 3)
            # If scores are very close, this nudges towards the persona.
            if score < best_score:
                best_score = score
                best_slot = slot
            elif score == best_score:
                # Tie-breaker: If costs are equal, prefer later slots (2, 3) over earlier ones (0, 1)
                if slot > best_slot and best_slot < 2:
                    best_slot = slot
        
        recommendations[day_name] = best_slot

    return recommendations

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load scenario data
    # NOTE: In a real execution environment, scenario.json is in the same directory.
    fixed, daily, day_names = load_scenario_data()

    # For local testing/structure validation, we must inject the neighbor data directly
    # as it is not guaranteed to be in scenario.json structure.
    neighbor_examples_data = [
        {
            "Neighbor 4": [
                {
                    "Base demand": [0.90, 0.60, 0.70, 0.80], 
                    "Comfort penalty": 0.16, 
                    "Preferred slots": [0, 3], 
                    "ground_truth min-cost slots by day": [3, 3, 3, 2, 3, 3, 3]
                ]
            ]
        },
        {
            "Neighbor 1": [
                {
                    "Base demand": [1.20, 0.70, 0.80, 0.60], 
                    "Comfort penalty": 0.18, 
                    "Preferred slots": [0, 2], 
                    "ground_truth min-cost slots by day": [0, 1, 2, 0, 1, 2, 0]
                ]
            ]
        }
    ]
    
    # Reorder day names to match the simulation structure (Day 1 to Day 7)
    # The loaded day_names should already be in order based on the prompt structure.
    
    # 2. Determine recommendation
    if fixed and daily and day_names:
        slot_recommendations = determine_recommendation(
            fixed, 
            daily, 
            day_names, 
            neighbor_examples_data
        )

        # 3. Format output
        # Ensure output is strictly ordered by Day 1, Day 2, ..., Day 7
        final_output_list = []
        for day_name in day_names:
            if day_name in slot_recommendations:
                final_output_list.append(slot_recommendations[day_name])
            else:
                # Fallback: if a day is missing, default to slot 0 or the last computed slot
                final_output_list.append(0) 

        # 4. Write global_policy_output.json
        output_filename = "global_policy_output.json"
        with open(output_filename, 'w') as f:
            json.dump(final_output_list, f, indent=4)

        # print(f"Policy decisions saved to {output_filename}: {final_output_list}")
        
    else:
        # Critical failure in data loading, create a safe default output
        print("Data loading failed. Outputting default policy: [3, 3, 3, 3, 3, 3, 3]")
        default_output = [3] * 7
        with open("global_policy_output.json", 'w') as f:
            json.dump(default_output, f, indent=4)
        # The policy script should only contain the code required to run, so we ensure the final output is just the script.
        pass # In the final output, this block is omitted.
        
# The script ends here, providing only the code above.