import json
import numpy as np

# --- Input Data ---
# Hardcoding the input data as per instructions, since file loading isn't strictly required if data is present in the context block.
SCENARIO_DATA = {
    "scenario_id": "ev_peak_sharing_1",
    "slots": {0: "19-20", 1: "20-21", 2: "21-22", 3: "22-23"},
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": 6.8,
    "baseline_load": [5.2, 5.0, 4.9, 6.5],
    "slot_min_sessions": {0: 1, 1: 1, 2: 1, 3: 1},
    "slot_max_sessions": {0: 2, 1: 2, 2: 1, 3: 2},
    "spatial_carbon": {
        1: [440, 460, 490, 604],
        2: [483, 431, 471, 600],
        3: [503, 473, 471, 577],
        4: [617, 549, 479, 363],
        5: [411, 376, 554, 623]
    },
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": {1: [330, 520, 560, 610], 2: [550, 340, 520, 600], 3: [590, 520, 340, 630], 4: [620, 560, 500, 330], 5: [360, 380, 560, 620]}},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": {1: [510, 330, 550, 600], 2: [540, 500, 320, 610], 3: [310, 520, 550, 630], 4: [620, 540, 500, 340], 5: [320, 410, 560, 640]}},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": {1: [540, 500, 320, 600], 2: [320, 510, 540, 600], 3: [560, 330, 520, 610], 4: [620, 560, 500, 330], 5: [330, 420, 550, 640]}},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": {1: [320, 520, 560, 600], 2: [550, 330, 520, 580], 3: [600, 540, 500, 320], 4: [560, 500, 330, 540], 5: [500, 340, 560, 630]}},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": {1: [510, 330, 560, 600], 2: [560, 500, 320, 590], 3: [320, 520, 540, 620], 4: [630, 560, 510, 340], 5: [330, 420, 560, 630]}},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": {1: [540, 500, 320, 610], 2: [320, 510, 560, 620], 3: [560, 340, 520, 610], 4: [640, 560, 510, 330], 5: [520, 330, 540, 600]}},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": {1: [330, 520, 560, 610], 2: [540, 330, 520, 600], 3: [580, 540, 330, 620], 4: [630, 560, 500, 330], 5: [520, 330, 550, 600]}},
    }
}

# --- Persona & Constraints ---
MY_BASE_DEMAND = np.array([1.20, 0.70, 0.80, 0.60])
TOTAL_BASE_DEMAND = np.sum(MY_BASE_DEMAND) # 3.3
MY_LOCATION = 1
PREFERRED_SLOTS = [0, 2]
SLOT_MAX_SESSIONS = [2, 2, 1, 2] # Slot 2 restricted to 1 session

# --- Strategy ---
# Goal: Meet 3.3 total demand. Prioritize low Price & low Carbon. Strongly prefer slots 0 and 2.

daily_recommendations = []
day_names = list(SCENARIO_DATA["days"].keys())

for day_name in day_names:
    day_data = SCENARIO_DATA["days"][day_name]

    tariffs = np.array(day_data["Tariff"])
    carbons = np.array(day_data["Carbon"])

    # 1. Scoring: Combine price (weight 0.6) and carbon (weight 0.4). Lower is better.
    
    # Normalize inputs for scoring (0=best, 1=worst)
    # Add a tiny epsilon to the range denominator to prevent division by zero if max==min
    eps = 1e-6
    range_price = np.max(tariffs) - np.min(tariffs) + eps
    range_carbon = np.max(carbons) - np.min(carbons) + eps
    
    norm_price = (tariffs - np.min(tariffs)) / range_price
    norm_carbon = (carbons - np.min(carbons)) / range_carbon

    score = 0.6 * norm_price + 0.4 * norm_carbon
    
    # Initialize priority based on score (higher score = higher priority to load)
    # Since score is normalized 0..1 (low is good), we use (1 - score) * K as the loading potential/priority
    
    # 2. Define Priority based on Score, Preference, and Special Constraints
    
    priority = (1 - score) * 100 # Base priority proportional to how good the slot is (0-100)

    # Boost preferred slots (0 and 2)
    priority[0] += 20
    priority[2] += 20

    # Apply Day 6 special constraint: Slot 2 is rationed heavily
    if day_name == "Day 6":
        priority[2] *= 0.2 # Severe reduction in priority for slot 2

    # 3. Define hard utilization caps (max_util <= 1.0)
    max_util = np.ones(4) 
    
    # Slot 2 has max 1 session vs. others with max 2 sessions. Assume this implies a lower utilization limit.
    max_util[2] = 0.9 # Slight constraint on slot 2 utilization cap
    
    if day_name == "Day 6":
        max_util[2] = 0.4 # Heavy restriction due to rationing advisory

    # 4. Calculate allocation distribution based on priority, aiming for a total of 3.3
    
    total_priority = np.sum(priority)
    
    # Proportional allocation based on weighted priority
    if total_priority > 0:
        allocation = (priority / total_priority) * TOTAL_BASE_DEMAND
    else:
        # Fallback: spread evenly if calculation fails (should not happen)
        allocation = np.ones(4) * (TOTAL_BASE_DEMAND / 4.0)
        
    final_usage = np.clip(allocation, 0.0, max_util)

    # 5. Handle demand deficit (if clipping occurred)
    current_sum = np.sum(final_usage)
    remaining_demand = TOTAL_BASE_DEMAND - current_sum

    if remaining_demand > 1e-4:
        # Redistribute remaining charge into available headroom in non-capped slots
        headroom = max_util - final_usage
        total_headroom = np.sum(headroom)

        if total_headroom > 1e-6:
            adjustment_factor = remaining_demand / total_headroom
            adjustment = headroom * adjustment_factor
            final_usage += adjustment
        
    # Final clipping and application of minimum usage (mapping min_sessions=1 to min_usage=0.1)
    MIN_UTIL = np.array([0.1] * 4)
    
    final_usage = np.clip(final_usage, MIN_UTIL, 1.0)
    
    # Final normalization/clipping to ensure sum is close to 3.3 while respecting [0, 1]
    # If sum > 3.3, we scale down slightly, prioritizing high usage slots.
    final_sum = np.sum(final_usage)
    if final_sum > TOTAL_BASE_DEMAND + 1e-4:
        scaling_factor = TOTAL_BASE_DEMAND / final_sum
        final_usage *= scaling_factor
        
    # Final clip again just in case scaling pushed something slightly over 1.0
    final_usage = np.clip(final_usage, 0.0, 1.0)

    daily_recommendations.append(final_usage.tolist())

# --- Output Generation ---
OUTPUT_FILENAME = "local_policy_output.json"
policy_output = daily_recommendations

# Write to file
with open(OUTPUT_FILENAME, 'w') as f:
    json.dump(policy_output, f, indent=4)

# Since the output must be *only* runnable Python code that performs the task:
# We remove print statements and rely solely on the JSON writing mechanism.

# Final check on required output format: list of seven usage vectors (floats 0-1). This is met by daily_recommendations.
# Note: The instruction asks to write JSON list of seven *integers* (0-3) in the final constraint, but the operational requirement states floats in [0, 1]. Adhering to the operational requirement (floats in [0, 1]) as neighbor examples also show floats. The 'integers (0-3)' constraint seems erroneous given the [0, 1] usage vector requirement.
# Reverting to float output as planned.
# The instruction states: "writes a JSON list of seven integers (0-3) to 'local_policy_output.json'". This contradicts step 4: "Construct a list of 7 usage vectors, ensuring each element is in [0, 1]".
# Given the context of EV charging usage, floats in [0, 1] are the correct representation of continuous usage. We prioritize the usage vector definition over the contradictory integer constraint.

# If integer output (0, 1, 2, or 3) was strictly required, the strategy would be to quantize the final usage to the nearest load bin corresponding to 0, 0.33, 0.66, 1.0 utilization, but this loses precision required by the prompt structure. Sticking to float output [0, 1].
# --- End of code generation ---