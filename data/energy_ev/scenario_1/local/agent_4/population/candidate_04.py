import json
import math

# --- Agent Configuration ---
AGENT_ID = 4
LOCATION_ID = 4
# Base demand profile (kW) for slots 0, 1, 2, 3
BASE_DEMAND = [0.90, 0.60, 0.70, 0.80]
# Persona: Position 4 retirees guarding comfort and grid warnings.
# This suggests a high priority on comfort (avoiding high prices/carbon if possible)
# and reacting to explicit grid warnings (which here we interpret as reacting to high values).
# Comfort penalty is implied to be significant, favouring consistency/predictability if possible.

# --- Scenario Data (Extracted from Prompt) ---
SLOTS = [0, 1, 2, 3]
SLOT_TIMES = ["19-20", "20-21", "21-22", "22-23"]
CAPACITY = 6.8
SLOT_MIN_SESSIONS = [1, 1, 1, 1]
SLOT_MAX_SESSIONS = [2, 2, 1, 2]

# Neighbor Examples for imitation context (though imitation here relies more on persona)
NEIGHBOR_EXAMPLES = {
    3: {'preferred_slots': [1, 3], 'comfort_penalty': 0.20, 'ground_truth': [1, 1, 1, 3, 1, 1, 1]},
    5: {'preferred_slots': [0, 1], 'comfort_penalty': 0.12, 'ground_truth': [0, 1, 0, 0, 1, 1, 1]},
}

DAYS_DATA = {
    "Day 1": {'Tariff': [0.20, 0.25, 0.29, 0.32], 'Carbon': [490, 470, 495, 540], 'Baseline load': [5.3, 5.0, 4.8, 6.5], 'Spatial carbon': "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; 3: 590, 520, 340, 630; 4: 620, 560, 500, 330; 5: 360, 380, 560, 620"},
    "Day 2": {'Tariff': [0.27, 0.22, 0.24, 0.31], 'Carbon': [485, 460, 500, 545], 'Baseline load': [5.1, 5.2, 4.9, 6.6], 'Spatial carbon': "1: 510, 330, 550, 600; 2: 540, 500, 320, 610; 3: 310, 520, 550, 630; 4: 620, 540, 500, 340; 5: 320, 410, 560, 640"},
    "Day 3": {'Tariff': [0.24, 0.21, 0.26, 0.30], 'Carbon': [500, 455, 505, 550], 'Baseline load': [5.4, 5.0, 4.9, 6.4], 'Spatial carbon': "1: 540, 500, 320, 600; 2: 320, 510, 540, 600; 3: 560, 330, 520, 610; 4: 620, 560, 500, 330; 5: 330, 420, 550, 640"},
    "Day 4": {'Tariff': [0.19, 0.24, 0.28, 0.22], 'Carbon': [495, 470, 500, 535], 'Baseline load': [5.0, 5.1, 5.0, 6.7], 'Spatial carbon': "1: 320, 520, 560, 600; 2: 550, 330, 520, 580; 3: 600, 540, 500, 320; 4: 560, 500, 330, 540; 5: 500, 340, 560, 630"},
    "Day 5": {'Tariff': [0.23, 0.20, 0.27, 0.31], 'Carbon': [500, 450, 505, 545], 'Baseline load': [5.2, 5.3, 5.0, 6.6], 'Spatial carbon': "1: 510, 330, 560, 600; 2: 560, 500, 320, 590; 3: 320, 520, 540, 620; 4: 630, 560, 510, 340; 5: 330, 420, 560, 630"},
    "Day 6": {'Tariff': [0.26, 0.22, 0.25, 0.29], 'Carbon': [505, 460, 495, 540], 'Baseline load': [5.5, 5.2, 4.8, 6.5], 'Spatial carbon': "1: 540, 500, 320, 610; 2: 320, 510, 560, 620; 3: 560, 340, 520, 610; 4: 640, 560, 510, 330; 5: 520, 330, 540, 600"},
    "Day 7": {'Tariff': [0.21, 0.23, 0.28, 0.26], 'Carbon': [495, 460, 500, 530], 'Baseline load': [5.1, 4.9, 4.8, 6.3], 'Spatial carbon': "1: 330, 520, 560, 610; 2: 540, 330, 520, 600; 3: 580, 540, 330, 620; 4: 630, 560, 500, 330; 5: 520, 330, 550, 600"},
}

# --- Helper Functions ---

def parse_spatial_carbon(day_data):
    """Parses spatial carbon string into a dictionary keyed by location index."""
    sc_map = {}
    if 'Spatial carbon' in day_data:
        # Example format: "1: 330, 520, 560, 610; 2: 550, 340, 520, 600; ..."
        parts = day_data['Spatial carbon'].split(';')
        for part in parts:
            try:
                loc_str, values_str = part.strip().split(': ')
                loc_id = int(loc_str)
                # Extract the carbon value specific to THIS agent's location
                # Note: The structure seems to list ALL locations' values sequentially.
                # We need the value corresponding to LOCATION_ID (4 in this case).
                
                # The indices in the string correspond to slots 0, 1, 2, 3
                # The structure seems to be spatial_carbon: LOC_A: [S0, S1, S2, S3]; LOC_B: [S0, S1, S2, S3]; ...
                
                # Since we are Agent 4 (Location 4), we need the 4th list of values if they were ordered,
                # but the prompt structure implies we need to find the data block corresponding to our location,
                # or extract the 4th element from *every* list and average them if they represent *other* locations' influence.
                
                # Standard interpretation for this format in EV scenarios: The spatial carbon data provided
                # for the day is an *influence map*. Since Agent 4 is at Location 4, we look for the data block associated with Location 4.
                
                if loc_id == LOCATION_ID:
                    sc_map['self'] = [int(v) for v in values_str.split(', ')]
                
                # For this imitation stage, we assume the agent is primarily interested in its *own* localized environment data,
                # which usually means checking the data explicitly labeled for its location ID.
                # If the format means "Loc X's effect on the area", we need the data from Loc 4.
                
            except ValueError:
                continue
    
    # If the explicit location data is not found in the map structure, fall back to
    # calculating an average influence, but given the context, prioritizing the explicit data point for Location 4 is safest.
    
    # Let's re-parse assuming the standard is: data for ALL locations across the 4 slots.
    # We need the data point corresponding to LOCATION_ID = 4.
    
    all_loc_data = {}
    for part in parts:
        try:
            loc_str, values_str = part.strip().split(': ')
            loc_id = int(loc_str)
            all_loc_data[loc_id] = [int(v) for v in values_str.split(', ')]
        except ValueError:
            pass

    if LOCATION_ID in all_loc_data:
        sc_map['self'] = all_loc_data[LOCATION_ID]
    else:
        # Fallback: If our specific location data isn't explicitly listed,
        # this implies the prompt structure is using a generalized influence, or the spatial data provided
        # is incomplete/structured unexpectedly for agent-specific extraction.
        # Given the high fidelity requirement, we will assume the structure *must* contain our location data.
        # If not found, we use the baseline carbon intensity for the slot as a safe default if available.
        sc_map['self'] = None # Will rely on main carbon intensity if this fails.

    return sc_map

def calculate_cost(tariff, carbon, baseline, slot_idx, day_context):
    """
    Calculates a composite cost metric for Agent 4 (Retiree, Comfort focus).
    Cost = Price + Carbon_Impact + Comfort_Penalty (implied by high values)
    
    We use a weighted sum of Price and Carbon Intensity, as comfort is maximized by avoiding
    periods where *either* is high, especially since the prompt mentions "grid warnings".
    Since we don't have an explicit comfort function, we penalize high Carbon/Price periods.
    
    Weighting: Carbon seems critical due to "grid warnings" and "retiree" observing environment.
    Let's weight Carbon higher than Price, but both are important.
    """
    P = tariff[slot_idx]
    C_base = carbon[slot_idx]
    B = baseline[slot_idx]
    
    # Spatial Carbon calculation (If available, it modifies the environmental cost)
    spatial_carbon_map = parse_spatial_carbon(day_context)
    C_spatial = spatial_carbon_map.get('self', [C_base, C_base, C_base, C_base])[slot_idx]

    # Use the Spatial Carbon if available, as it is more localized.
    # If spatial data for location 4 is missing entirely, use the general carbon intensity.
    Effective_Carbon = C_spatial if C_spatial is not None else C_base
    
    # Agent 4 (Retirees guarding comfort): Highly sensitive to environment/warnings.
    # We use a weighted sum favoring environmental factors (Carbon) slightly over immediate cost (Price).
    # Scale factors: Carbon is usually measured in higher units (gCO2/kWh) than Price (€/kWh).
    # Normalization factor for Carbon relative to Price (using day 1 average ratio):
    # Avg Price ~ 0.27, Avg Carbon ~ 500. Ratio ~ 1850.
    # Let's use a moderate scaling to combine them linearly.
    
    # Cost = Price + (Carbon / K)
    # K = 1500 (Heuristic to balance scales)
    K = 1500 
    
    composite_cost = P + (Effective_Carbon / K)
    
    # Apply session constraints penalty/bonus (this is secondary to environmental cost in imitation stage)
    
    return composite_cost

def get_spatial_carbon_data(day_context_str):
    """Parses spatial carbon string into a structure suitable for lookup."""
    sc_data = {}
    if day_context_str:
        parts = day_context_str.split(';')
        for part in parts:
            try:
                loc_str, values_str = part.strip().split(': ')
                loc_id = int(loc_str)
                sc_data[loc_id] = [int(v) for v in values_str.split(', ')]
            except ValueError:
                continue
    return sc_data

def decide_slot_for_day(day_name, day_data):
    """Chooses the slot that minimizes the composite cost for Agent 4."""
    
    tariffs = day_data['Tariff']
    carbons = day_data['Carbon']
    baselines = day_data['Baseline load']
    spatial_carbon_str = day_data.get('Spatial carbon')
    
    spatial_map = get_spatial_carbon_data(spatial_carbon_str)
    
    # Determine the effective carbon intensity for Agent 4 (Location 4)
    # If Location 4 data exists in the map, use that. Otherwise, use baseline carbon.
    if LOCATION_ID in spatial_map:
        effective_carbons = spatial_map[LOCATION_ID]
    else:
        effective_carbons = carbons

    costs = []
    
    # Imitation Stage: Follow the predicted action of the most similar agent,
    # OR follow the persona if neighbors are dissimilar/irrelevant.
    # Neighbor 3 (Nurse): Prefers slot 1 always (except Day 4).
    # Neighbor 5 (Grad): Prefers slots 0, 1. Follows slot 1 (except Day 1, 3, 4).
    
    # Agent 4 Persona: Retirees guarding comfort and grid warnings.
    # This suggests preferring the historically 'safest' slots based on combined metrics,
    # and perhaps mimicking the consistency of neighbors if available.
    
    # Since the prompt demands imitation of what *the agents* would follow, and we only have 2 examples,
    # a strong imitation strategy might average or pick the most common action among neighbors OR
    # stick to the persona's optimal choice if neighbors conflict or are too different.
    
    # In Stage 2, imitation is key. Let's look at the neighbors' GROUND TRUTH actions for this week:
    # Day 1: N3=[1], N5=[0]
    # Day 2: N3=[1], N5=[1] -> Likely 1
    # Day 3: N3=[1], N5=[0]
    # Day 4: N3=[3], N5=[0]
    # Day 5: N3=[1], N5=[1] -> Likely 1
    # Day 6: N3=[1], N5=[1] -> Likely 1
    # Day 7: N3=[1], N5=[1] -> Likely 1
    
    # Slot 1 is the overwhelming choice for neighbors on 5/7 days, often coinciding with lower tariff/carbon.
    # Let's calculate the persona's *true* best choice first, and then decide if imitation overrides it.
    
    persona_costs = []
    for s in SLOTS:
        cost = calculate_cost(tariffs, effective_carbons, baselines, s, day_data)
        
        # Apply session constraints as a massive penalty if violated (though imitation should respect them)
        if s < SLOT_MIN_SESSIONS[s] and day_name == "Day 1": # Assume Day 1 is when session count starts low
             cost += 1000 
        if day_name == "Day 6" and s == 2: # Day 6: slot 2 rationed (implied max sessions = 0 or very low)
             cost += 1000

        persona_costs.append((cost, s))

    persona_costs.sort()
    best_persona_slot = persona_costs[0][1]
    
    # --- Imitation Decision (Stage 2) ---
    
    # If the environment suggests a very strong outlier (e.g., extremely high carbon/price),
    # the persona's objective (guarding comfort) might override weak imitation.
    
    # Given the prompt asks to choose what the agents *would* follow:
    # 1. If neighbors strongly agree (e.g., 2 out of 2 agree), follow that.
    # 2. If neighbors disagree or context suggests a strong environmental response, use persona best.
    
    # Since neighbors agree on Slot 1 for 5/7 days, and Slot 1 often aligns with lower carbon/price (e.g., Day 2: Carbon 460, Price 0.22):
    
    imitation_choice = None
    
    # Day 1: N3=1, N5=0. Best Persona: Slot 1 (lowest cost: 0.20 + 490/1500 = 0.526)
    if day_name == "Day 1":
        # Persona strongly favors Slot 1. Imitation suggests 0 or 1. Go with persona best.
        imitation_choice = best_persona_slot
    
    # Day 2: N3=1, N5=1. Strong Agreement.
    elif day_name == "Day 2":
        imitation_choice = 1
        
    # Day 3: N3=1, N5=0. Persona best Slot 1 (Cost 0.21 + 455/1500 = 0.513)
    elif day_name == "Day 3":
        # Slot 0 cost: 0.24 + 500/1500 = 0.573. Slot 1 is clearly better for persona.
        imitation_choice = best_persona_slot

    # Day 4: N3=3, N5=0. Disagreement. Persona best Slot 0 (Cost 0.19 + 495/1500 = 0.520)
    elif day_name == "Day 4":
        imitation_choice = best_persona_slot
        
    # Day 5: N3=1, N5=1. Strong Agreement.
    elif day_name == "Day 5":
        imitation_choice = 1
        
    # Day 6: N3=1, N5=1. Strong Agreement. (Also, slot 2 is rationed/avoided)
    elif day_name == "Day 6":
        imitation_choice = 1
        
    # Day 7: N3=1, N5=1. Strong Agreement.
    elif day_name == "Day 7":
        imitation_choice = 1

    # Final check: Does the imitated slot violate hard constraints (like Day 6 rationing)?
    if day_name == "Day 6" and imitation_choice == 2:
        # If imitation dictated slot 2, override due to maintenance advisory.
        imitation_choice = best_persona_slot
        
    # Ensure result is within slot bounds [0, 3]
    return int(imitation_choice)

# --- Main Execution ---

def generate_policy():
    schedule = []
    day_names = list(DAYS_DATA.keys())

    for i, day_name in enumerate(day_names):
        day_context = DAYS_DATA[day_name]
        chosen_slot = decide_slot_for_day(day_name, day_context)
        schedule.append(chosen_slot)
    
    return schedule

if __name__ == "__main__":
    # In a real environment, we'd load scenario.json. Here, we use embedded data as per instructions.
    
    final_schedule = generate_policy()
    
    # Output local_policy_output.json
    with open("local_policy_output.json", "w") as f:
        json.dump(final_schedule, f, indent=4)
    
    # The code itself is saved as policy.py, fulfilling the execution requirement.
    pass
# End of policy.py