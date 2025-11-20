import json
import numpy as np
import os
from datetime import datetime

# --- Configuration Constants based on Agent Profile ---
AGENT_ID = 3
SLOTS = [0, 1, 2, 3]  # 19-20, 20-21, 21-22, 22-23
CAPACITY = 6.8
BASE_DEMAND = np.array([0.60, 0.80, 0.90, 0.70])
NEIGHBOR_EXAMPLES = [
    {
        "id": 2,
        "location": 2,
        "base_demand": np.array([0.70, 1.00, 0.80, 0.50]),
        "preferred_slots": [1, 2],
        "comfort_penalty": 0.14,
        "ground_truth": [1, 2, 0, 1, 2, 0, 1]
    },
    {
        "id": 5,
        "location": 5,
        "base_demand": np.array([0.50, 0.70, 0.60, 0.90]),
        "preferred_slots": [0, 1],
        "comfort_penalty": 0.12,
        "ground_truth": [0, 0, 0, 0, 0, 1, 1]
    }
]
ALPHA = 40.00  # Weight for Carbon
BETA = 0.50    # Weight for Price
GAMMA = 12.00  # Weight for Congestion (Spatial Carbon)

# --- Policy Parameters for Agent 3 (Position 3, Nurse, Location 3) ---
# Nurse on night shift suggests high demand late (0.90 in slot 2), maybe a need to charge before/after shift.
# Location 3 has specific spatial carbon patterns.
# Preferred slots based on base demand: Slot 2 (0.90) is peak personal use.
# Temporal preference: Avoid the highest price/carbon slot (Slot 3: 0.30/750) if possible, but Slot 2 is high base demand.
# Given location 3: Spatial carbon values are generally lower in slot 1 and 2 on several days.

# Agent 3's Comfort/Preference: Prioritize slots that align with base demand, especially slot 2, but slot 1 is also strong.
# Penalize slots that conflict with high base demand unless external incentives are very strong.
# Let's assign a lower comfort penalty to slots matching high base demand (Slot 2 > Slot 1 > Slot 0 > Slot 3)
COMFORT_PREFERENCE = {
    0: 1.0,  # Mildly preferred
    1: 1.2,  # Preferred
    2: 1.5,  # Most preferred (highest base demand)
    3: 0.8   # Least preferred (lowest base demand)
}

# Spatial Carbon Weighting (Location 3 specific)
# Location 3 spatial carbon is often lowest in Slot 1 or 2 during the reference period.
SPATIAL_WEIGHT_BASE = 1.0


def load_scenario_data(base_path="."):
    """Loads scenario data from JSON files relative to the execution directory."""
    try:
        # Assuming scenario.json is in the same directory or one level up if running from a specific agent subfolder structure
        scenario_path = os.path.join(base_path, "scenario.json")
        if not os.path.exists(scenario_path):
            # Fallback if the file structure expects the data file to be relative to the context
            # In a typical ECHO setup, the agent execution directory usually contains the needed files.
            scenario_path = "scenario.json"

        with open(scenario_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: scenario.json not found at expected path.")
        # Create dummy data structure matching expectations if file loading fails, for local testing structure verification
        return {
            "scenario_id": "ev_peak_sharing_1",
            "slots": "0: 19-20, 1: 20-21, 2: 21-22, 3: 22-23",
            "price": [0.23, 0.24, 0.27, 0.30],
            "carbon_intensity": [700, 480, 500, 750],
            "capacity": 6.8,
            "baseline_load": [5.2, 5.0, 4.9, 6.5],
            "slot_min_sessions": [1, 1, 1, 1],
            "slot_max_sessions": [2, 2, 1, 2],
            "spatial_carbon": "1: 440, 460, 490, 604 | 2: 483, 431, 471, 600 | 3: 503, 473, 471, 577 | 4: 617, 549, 479, 363 | 5: 411, 376, 554, 623",
            "days": {
                "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; 3: 590, 520, 340, 630; 4: 620, 560, 500, 330; 5: 360, 380, 560, 620"},
                "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": "1: 510, 330, 550, 600; 2: 540, 500, 320, 610; 3: 310, 520, 550, 630; 4: 620, 540, 500, 340; 5: 320, 410, 560, 640"},
                "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": "1: 540, 500, 320, 600; 2: 320, 510, 540, 600; 3: 560, 330, 520, 610; 4: 620, 560, 500, 330; 5: 330, 420, 550, 640"},
                "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": "1: 320, 520, 560, 600; 2: 550, 330, 520, 580; 3: 600, 540, 500, 320; 4: 560, 500, 330, 540; 5: 500, 340, 560, 630"},
                "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": "1: 510, 330, 560, 600; 2: 560, 500, 320, 590; 3: 320, 520, 540, 620; 4: 630, 560, 510, 340; 5: 330, 420, 560, 630"},
                "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": "1: 540, 500, 320, 610; 2: 320, 510, 560, 620; 3: 560, 340, 520, 610; 4: 640, 560, 510, 330; 5: 520, 330, 540, 600"},
                "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": "1: 330, 520, 560, 610; 2: 540, 330, 520, 600; 3: 580, 540, 330, 620; 4: 630, 560, 500, 330; 5: 520, 330, 550, 600"}
            }
        }
    except Exception as e:
        print(f"Critical error loading scenario data: {e}")
        exit(1)


def parse_spatial_carbon(spatial_str, agent_location, num_slots=4):
    """Parses the spatial carbon string for the agent's location."""
    sc_data = {}
    parts = spatial_str.split('|')
    for part in parts:
        if ':' in part:
            loc_id_str, values_str = part.strip().split(':')
            loc_id = int(loc_id_str)
            values = [int(v) for v in values_str.split(',')]
            sc_data[loc_id] = values

    # Extract data for the agent's location (Agent 3 is Location 3)
    if agent_location in sc_data and len(sc_data[agent_location]) == num_slots:
        return np.array(sc_data[agent_location])
    
    # Fallback if parsing fails or data is missing for this location
    return np.zeros(num_slots)


def calculate_objective_score(day_data, slot_idx, agent_demand, neighbor_sessions, historical_sessions):
    """
    Calculates a multi-objective score for a given slot.
    Score = ALPHA * Carbon + BETA * Price + GAMMA * Congestion_Cost + Comfort_Penalty
    Lower score is better.
    """
    price = day_data['Tariff'][slot_idx]
    carbon = day_data['Carbon'][slot_idx]
    spatial_carbon = day_data['Spatial_Carbon_Agent'][slot_idx]
    baseline = day_data['Baseline_load'][slot_idx]
    
    # --- 1. Global Objectives (Normalized/Scaled) ---
    
    # Carbon Score: Use raw carbon intensity as it's scaled by ALPHA
    carbon_score = ALPHA * carbon

    # Price Score: Use raw price as it's scaled by BETA
    price_score = BETA * price

    # Congestion Score (Spatial Carbon): Heavily penalize slots where local neighbors are already constrained
    # We use the agent's own spatial carbon as a proxy for local impact/congestion sensitivity.
    # Weight by GAMMA
    congestion_score = GAMMA * spatial_carbon
    
    # --- 2. Agent Preference (Comfort/Utility) ---
    
    # Comfort Penalty: Inverse of preference (Higher preference = Lower penalty)
    comfort_val = COMFORT_PREFERENCE.get(slot_idx, 1.0)
    # Since COMFORT_PREFERENCE is a utility measure (higher is better), the penalty is 1/Utility
    comfort_penalty = 100.0 / comfort_val # Scale penalty high enough to matter

    # --- 3. Local Demand Contribution (Implicitly handled by COMFORT_PREFERENCE, but verify load) ---
    
    # Ensure capacity is respected (though capacity constraints are typically handled post-selection, 
    # we use this to bias against slots likely to overload if historical data suggests it).
    
    # --- Total Score (Lower is better) ---
    total_score = carbon_score + price_score + congestion_score + comfort_penalty
    
    return total_score


def determine_coordination_target(day_name, day_data, neighbor_examples):
    """
    Determines an optimal slot selection for Agent 3 based on goals and observed neighbor behavior.
    The core coordination strategy is to avoid slots heavily used by neighbors, especially if those slots
    have high carbon intensity or if neighbors have hard constraints (like low max sessions).
    """
    
    # 1. Determine available neighbor sessions for this day
    neighbor_sessions = {} # {slot_idx: count}
    for neighbor in neighbor_examples:
        # Ground truth usually holds the chosen slot index for the day
        day_index = list(day_data['Days'].keys()).index(day_name)
        if day_index < len(neighbor['ground_truth']):
            chosen_slot = neighbor['ground_truth'][day_index]
            neighbor_sessions[chosen_slot] = neighbor_sessions.get(chosen_slot, 0) + 1
            
    # 2. Calculate Scores for all slots
    day_scores = {}
    for s in SLOTS:
        # Assume minimum required demand for scoring purposes
        score = calculate_objective_score(
            day_data, s, BASE_DEMAND[s], neighbor_sessions, None
        )
        day_scores[s] = score

    # 3. Apply coordination modification based on neighbor usage
    # Strategy: If a slot is heavily used by neighbors (e.g., 1 or more), slightly penalize it unless 
    # the slot is extremely favorable for Agent 3 (i.e., the base score is very low).
    
    MODIFIED_SCORES = day_scores.copy()
    
    # Find the best score among neighbors' choices
    neighbor_choices = [n['ground_truth'][list(day_data['Days'].keys()).index(day_name)] 
                        for n in neighbor_examples 
                        if list(day_data['Days'].keys()).index(day_name) < len(n['ground_truth'])]
    
    NEIGHBOR_PENALTY_MULTIPLIER = 1.05 # 5% increase in cost if a slot is a neighbor's known preference/choice

    for s in SLOTS:
        if s in neighbor_choices:
            # Increase cost slightly for coordination, unless this slot is vastly superior (e.g., massive carbon drop)
            original_score = day_scores[s]
            
            # Only apply penalty if the base score isn't already overwhelmingly low (e.g., less than 1/10th of the next best)
            sorted_scores = sorted(day_scores.values())
            if len(sorted_scores) > 1 and original_score < sorted_scores[1] / 10.0:
                 # Slot is drastically better than alternatives, don't penalize heavily
                 pass
            else:
                MODIFIED_SCORES[s] = original_score * NEIGHBOR_PENALTY_MULTIPLIER

    # 4. Select the slot with the minimum modified score
    best_slot = min(MODIFIED_SCORES, key=MODIFIED_SCORES.get)
    
    # 5. Final check against Min/Max sessions (Capacity proxy)
    # Since we don't know the exact number of neighbors, we use the neighbor examples as a guide.
    # If neighbor load is high in a slot, and this slot is at max capacity for the entire system (slot_max_sessions=1 for one neighbor), 
    # we might prefer another slot, unless the best slot is strongly favored.
    
    # For this agent (Agent 3), slot_max_sessions for slot 2 is 1. Slot 0, 1, 3 are 2.
    # If the globally best slot according to score has a restrictive max session (like slot 2 has max 1), 
    # and many neighbors chose it (which we don't track perfectly here, relying only on ground truth),
    # we proceed with the minimum score unless the score difference is negligible.
    
    # Since we prioritize the calculated score which incorporates carbon/price/comfort, we stick to the minimum score result.
    
    return best_slot


def policy():
    scenario = load_scenario_data()
    
    # Global parameters from scenario header (used for context, though ALPHA/BETA/GAMMA are hardcoded)
    global_price = np.array(scenario['price'])
    global_carbon = np.array(scenario['carbon_intensity'])
    global_baseline = np.array(scenario['baseline_load'])
    slot_max_sessions = scenario['slot_max_sessions']
    
    # Parse spatial carbon data (Location 3 is Agent 3's location)
    agent_location = AGENT_ID # Agent 3 is located at position 3
    spatial_carbon_map = {}
    for day_name, day_data in scenario['days'].items():
        spatial_carbon_map[day_name] = parse_spatial_carbon(
            day_data['Spatial carbon'], agent_location
        )

    # Prepare structured data for iteration
    day_iterations = []
    day_names = list(scenario['days'].keys())
    for i, day_name in enumerate(day_names):
        day_data = scenario['days'][day_name]
        day_iterations.append({
            'name': day_name,
            'Tariff': np.array(day_data['Tariff']),
            'Carbon': np.array(day_data['Carbon']),
            'Baseline_load': np.array(day_data['Baseline load']),
            'Spatial_Carbon_Agent': spatial_carbon_map[day_name],
            'Days': scenario['days'] # Keep reference to full dict for iteration index tracking
        })

    recommendations = []
    
    # We need to simulate or estimate neighbor sessions for coordination.
    # Since we are only allowed to use neighbor *examples* (ground truth), we will use the ground truth 
    # of the neighbors for the *current* day to inform coordination bias.
    
    for day_data in day_iterations:
        day_name = day_data['name']
        
        # For coordination, we pass the ground truth of the neighbors for this specific day index
        # (This assumes the simulation environment provides the 'ground truth' historical context for coordination)
        
        recommended_slot = determine_coordination_target(day_name, day_data, NEIGHBOR_EXAMPLES)
        recommendations.append(recommended_slot)

    # --- Output Generation ---
    
    # 1. Create global_policy_output.json structure
    output_data = {
        "agent_id": AGENT_ID,
        "scenario_id": scenario['scenario_id'],
        "recommendations": [
            {"day": day_names[i], "slot": recommendations[i]} for i in range(len(day_names))
        ],
        "policy_parameters_used": {
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
            "comfort_preference": COMFORT_PREFERENCE
        }
    }
    
    with open("global_policy_output.json", "w") as f:
        json.dump(output_data, f, indent=4)

    # 2. Create policy.py (This script itself serves as policy.py, so we just ensure it runs)
    # The instructions require the script to save the JSON, which we have done.

if __name__ == "__main__":
    policy()