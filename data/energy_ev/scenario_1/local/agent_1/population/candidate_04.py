import json
import numpy as np

# --- Scenario Data Simulation (Hardcoded based on prompt context) ---
SLOTS = [
    {"start": 19, "end": 20, "price": 0.23, "carbon": 700, "baseline": 5.2},
    {"start": 20, "end": 21, "price": 0.24, "carbon": 480, "baseline": 5.0},
    {"start": 21, "end": 22, "price": 0.27, "carbon": 500, "baseline": 4.9},
    {"start": 22, "end": 23, "price": 0.30, "carbon": 750, "baseline": 6.5},
]
CAPACITY = 6.8
SLOT_MIN_SESSIONS = [1, 1, 1, 1]
SLOT_MAX_SESSIONS = [2, 2, 1, 2]

DAY_DATA = {
    "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5]},
    "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6]},
    "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4]},
    "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7]},
    "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6]},
    "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5]},
    "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3]},
}

# --- Agent Profile ---
MY_LOCATION = 1
MY_BASE_DEMAND = [1.20, 0.70, 0.80, 0.60]

# --- Neighbor Examples (For Imitation Context) ---
# Neighbor 2 (Location 2): Prioritizes early slots (1, 2) for headroom. Ground truth strongly favors slot 1.
# Neighbor 3 (Location 3): Night-shift nurse. Prefers 1, 3. Ground truth strongly favors slot 1.

# --- Agent 1 Persona Analysis (Battery Engineer balancing budget and solar backfeed) ---
# Goal: Balance budget (low price) and solar backfeed (low carbon, likely correlated with solar availability).
# As a battery engineer focused on backfeed, slots with low carbon intensity are highly desirable, especially if they align with low cost.
# The forecast shows the lowest carbon is Slot 1 (480), followed by Slot 2 (500).
# The lowest price is Slot 0 (0.23).

# Imitation Strategy (Stage 2):
# Observe neighbors' choices: Both neighbors strongly favor Slot 1 across almost all days (Day 1-7).
# Neighbor 2 (Headroom priority): [1, 1, 1, 1, 1, 1, 1]
# Neighbor 3 (Night Nurse): [1, 1, 1, 3, 1, 1, 1]
# The overwhelming trend is Slot 1 (20:00-21:00). This slot has the lowest carbon intensity in the forecast (480).
# The agent should imitate this strong preference, especially since Slot 1 is also the second cheapest in the forecast (0.24).

def calculate_cost(day_index, slot_index):
    """Calculates a weighted cost metric based on price and carbon for imitation."""
    day_key = list(DAY_DATA.keys())[day_index]
    tariff = DAY_DATA[day_key]["Tariff"][slot_index]
    carbon = DAY_DATA[day_key]["Carbon"][slot_index]
    
    # Weighting: Price is the primary budget concern, but carbon is explicitly mentioned in the persona.
    # We use a simple average or prioritize the lowest value if available, but for imitation,
    # we prioritize the slot that is consistently good for the observed pattern (Slot 1).
    
    # For robustness (if not strictly imitating slot 1), prioritize low carbon/price.
    # Since we are imitating, we use a penalty structure that rewards the most common neighbor choice (Slot 1).
    
    # Cost metric: Price + 0.5 * Carbon/1000 (to normalize carbon influence)
    return tariff + 0.5 * (carbon / 1000.0)

def choose_slot(day_index):
    day_name = list(DAY_DATA.keys())[day_index]
    
    # --- Imitation Logic ---
    # Observe the consistent behavior of neighbors: Slot 1 is overwhelmingly preferred.
    # This aligns well with Agent 1's underlying preference for low carbon (Slot 1 is lowest carbon in forecast).
    
    # Check neighbor ground truth again:
    # N2: [1, 1, 1, 1, 1, 1, 1]
    # N3: [1, 1, 1, 3, 1, 1, 1]
    
    # The imitation target is clearly Slot 1.
    imitation_slot = 1
    
    # Sanity Check: Does the imitation slot respect constraints?
    if SLOT_MIN_SESSIONS[imitation_slot] <= 1 <= SLOT_MAX_SESSIONS[imitation_slot]:
        # Also check if this slot is 'bad' according to the day's data (e.g., highest carbon/price)
        
        # Calculate neighbor consistency score for this day (How much do neighbors align on Slot 1 today?)
        # Since we don't have the exact neighbor choices for the *current* context, we stick to the observed historical pattern for Stage 2.
        return imitation_slot
    else:
        # Fallback: If the historical imitation target (1) is somehow constrained, fall back to a local optimization metric (e.g., lowest carbon in the forecast).
        # Forecast lowest carbon is Slot 1 (480), then Slot 2 (500).
        
        # For this specific scenario, Slot 1 is always constrained correctly (Min=1, Max=2). We will always return 1 based on strong imitation evidence.
        return 1


def generate_policy():
    policy = []
    
    # Iterate over the 7 days
    for day_index in range(7):
        chosen_slot = choose_slot(day_index)
        policy.append(chosen_slot)
        
    return policy

# --- Execution ---
output_policy = generate_policy()

# Create local_policy_output.json
output_data = {
    "agent_1": output_policy
}

with open("local_policy_output.json", "w") as f:
    json.dump(output_data, f, indent=4)

# This script will generate local_policy_output.json containing a 7-day slot plan based on imitating neighbors.
# Given the overwhelming evidence for Slot 1 in neighbor examples, the policy will be [1, 1, 1, 1, 1, 1, 1].
import json
import numpy as np

# --- Scenario Data Simulation (Hardcoded based on prompt context) ---
SLOTS = [
    {"start": 19, "end": 20, "price": 0.23, "carbon": 700, "baseline": 5.2},
    {"start": 20, "end": 21, "price": 0.24, "carbon": 480, "baseline": 5.0},
    {"start": 21, "end": 22, "price": 0.27, "carbon": 500, "baseline": 4.9},
    {"start": 22, "end": 23, "price": 0.30, "carbon": 750, "baseline": 6.5},
]
CAPACITY = 6.8
SLOT_MIN_SESSIONS = [1, 1, 1, 1]
SLOT_MAX_SESSIONS = [2, 2, 1, 2]

DAY_DATA = {
    "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5]},
    "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6]},
    "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4]},
    "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7]},
    "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6]},
    "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5]},
    "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3]},
}

# --- Agent Profile ---
MY_LOCATION = 1
MY_BASE_DEMAND = [1.20, 0.70, 0.80, 0.60]

# --- Neighbor Examples (For Imitation Context) ---
# N2 GT: [1, 1, 1, 1, 1, 1, 1]
# N3 GT: [1, 1, 1, 3, 1, 1, 1]
# Strong imitation target: Slot 1

def choose_slot(day_index):
    """In Stage 2 (Imitation), strictly follows the most common historical neighbor behavior."""
    
    # The most common, stable choice among neighbors (and one that aligns with Agent 1's low-carbon interest) is Slot 1.
    imitation_slot = 1
    
    # Constraints check (Slot 1 is always valid: Min 1, Max 2)
    if SLOT_MIN_SESSIONS[imitation_slot] <= 1 <= SLOT_MAX_SESSIONS[imitation_slot]:
        return imitation_slot
    else:
        # Should not happen for slot 1 in this scenario, but fallback to lowest forecast carbon (Slot 1)
        return 1


def generate_policy():
    policy = []
    
    # Iterate over the 7 days
    for day_index in range(7):
        chosen_slot = choose_slot(day_index)
        policy.append(chosen_slot)
        
    return policy

# --- Execution ---
output_policy = generate_policy()

# Create local_policy_output.json
output_data = {
    "agent_1": output_policy
}

with open("local_policy_output.json", "w") as f:
    json.dump(output_data, f, indent=4)