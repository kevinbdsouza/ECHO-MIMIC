import json
import numpy as np

# 1. Load scenario data and agent profile
try:
    with open('scenario.json', 'r') as f:
        scenario_data = json.load(f)
except FileNotFoundError:
    # Fallback for environments where the file might be one level up
    try:
        with open('../scenario.json', 'r') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        # In a constrained execution environment, we must assume the file is present
        # or raise an error if loading fails. For this output format, we proceed assuming success.
        pass

# Extract constants
SLOTS = scenario_data['slots']
N_SLOTS = len(SLOTS)
CAPACITY = scenario_data['capacity']
SLOT_MIN = [scenario_data['slot_min_sessions'][str(i)] for i in range(N_SLOTS)]
SLOT_MAX = [scenario_data['slot_max_sessions'][str(i)] for i in range(N_SLOTS)]

# Agent Profile (Position 1 battery engineer balancing budget and solar backfeed)
AGENT_LOCATION = 1
AGENT_BASE_DEMAND = np.array([1.20, 0.70, 0.80, 0.60])
AGENT_PREFERRED_SLOTS = [0, 2] # Derived from persona goal (Budget/Solar backfeed -> favoring early/mid slots)

# Parameters derived from global context
ALPHA = scenario_data['alpha']  # Weight for Carbon/Price trade-off
BETA = scenario_data['beta']    # Weight for Congestion/Capacity
GAMMA = scenario_data['gamma']  # Weight for comfort/preference

# --- Decision Logic ---
daily_policies = []
days_data = scenario_data['days']
day_keys = sorted(days_data.keys())

for day_key in day_keys:
    day_info = days_data[day_key]

    day_tariffs = np.array(day_info['Tariff'])
    day_carbons = np.array(day_info['Carbon'])
    day_baseline = np.array(day_info['Baseline load'])

    # Retrieve spatial carbon specific to Location 1
    spatial_carbon_str = [v for k, v in day_info['Spatial carbon'].items() if k == str(AGENT_LOCATION)][0]
    day_spatial_carbon = np.array([int(x.strip()) for x in spatial_carbon_str.split(', ')])

    # 1. Calculate Cost Components (to be MINIMIZED)

    # A. Carbon/Price Cost: Weighted combination of tariff and carbon intensity
    cost_carbon_objective = ALPHA * day_tariffs + day_carbons

    # B. Congestion Cost: Scaled baseline load (proxy for system stress)
    congestion_objective = day_baseline * BETA

    # C. Comfort/Preference Penalty (to be MINIMIZED)
    comfort_penalty = np.zeros(N_SLOTS)
    # Penalize deviation from preferred slots [0, 2]
    for i in range(N_SLOTS):
        if i not in AGENT_PREFERRED_SLOTS:
            comfort_penalty[i] += GAMMA 

    # D. Solar Incentive (to be MAXIMIZED, hence subtracted from minimization objective)
    # Strongly incentivize preferred slots [0, 2]
    solar_incentive = np.array([GAMMA * 2.0 if i in AGENT_PREFERRED_SLOTS else 0 for i in range(N_SLOTS)])

    # Total Score to Minimize (Lower is better)
    total_cost_to_minimize = (cost_carbon_objective + congestion_objective) - solar_incentive + comfort_penalty

    # --- Allocation Heuristic ---

    # Initialize usage based on minimum session requirements
    usage = np.array(SLOT_MIN, dtype=float)

    # Calculate scores for additional allocation above the minimums
    min_cost = np.min(total_cost_to_minimize)
    # Convert cost (to minimize) into utility (to maximize) for relative ranking
    normalized_utility_scores = min_cost - total_cost_to_minimize

    # Set utility for slots already at max capacity to zero/negative so they don't receive more load
    for i in range(N_SLOTS):
        if usage[i] >= SLOT_MAX[i]:
            normalized_utility_scores[i] = -np.inf

    # Focus on slots with positive relative utility
    positive_utility_scores = np.maximum(0, normalized_utility_scores)
    positive_utility_sum = np.sum(positive_utility_scores)

    if positive_utility_sum > 0:
        # Normalize scores to sum to 1 over the positive set
        relative_share = positive_utility_scores / positive_utility_sum

        # Distribute additional usage based on relative share, aiming to fill up to MAX capacity.
        
        for i in range(N_SLOTS):
            if relative_share[i] > 0:
                remaining_space = SLOT_MAX[i] - usage[i]

                # We use the relative share as the fractional multiplier of the remaining space.
                # Scale factor (e.g., 1.5) is an heuristic to encourage utilization when scores are good.
                fraction_to_use = min(1.0, relative_share[i] * 1.5) 

                additional_usage = fraction_to_use * remaining_space
                usage[i] = min(SLOT_MAX[i], usage[i] + additional_usage)

    # Final normalization and clipping (Usage must represent fraction of session time, 0 to 1.0)
    final_day_usage = np.clip(usage, 0.0, 1.0)

    # Re-enforce minimums just in case floating point arithmetic dropped below the required session count
    for i in range(N_SLOTS):
        if SLOT_MIN[i] > 0:
             final_day_usage[i] = max(final_day_usage[i], float(SLOT_MIN[i]))

    daily_policies.append(final_day_usage.tolist())

# 3. Write global_policy_output.json
output_data = {
    "policy": daily_policies
}

with open('global_policy_output.json', 'w') as f:
    json.dump(output_data, f, indent=4)