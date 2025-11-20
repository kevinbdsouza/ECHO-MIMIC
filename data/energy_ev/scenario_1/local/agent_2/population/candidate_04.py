import json

CONTEXT = {
    "scenario_id": "ev_peak_sharing_1",
    "slots": [0, 1, 2, 3],
    "price": [0.23, 0.24, 0.27, 0.30],
    "carbon_intensity": [700, 480, 500, 750],
    "capacity": 6.8,
    "baseline_load": [5.2, 5.0, 4.9, 6.5],
    "slot_min_sessions": [0, 1, 1, 1],
    "slot_max_sessions": [2, 2, 1, 2],
    "spatial_carbon": {
        "1": [440, 460, 490, 604], "2": [483, 431, 471, 600], "3": [503, 473, 471, 577],
        "4": [617, 549, 479, 363], "5": [411, 376, 554, 623]
    },
    "days": {
        "Day 1": {"Tariff": [0.20, 0.25, 0.29, 0.32], "Carbon": [490, 470, 495, 540], "Baseline load": [5.3, 5.0, 4.8, 6.5], "Spatial carbon": {"1": [330, 520, 560, 610], "2": [550, 340, 520, 600], "3": [590, 520, 340, 630], "4": [620, 560, 500, 330], "5": [360, 380, 560, 620]}},
        "Day 2": {"Tariff": [0.27, 0.22, 0.24, 0.31], "Carbon": [485, 460, 500, 545], "Baseline load": [5.1, 5.2, 4.9, 6.6], "Spatial carbon": {"1": [510, 330, 550, 600], "2": [540, 500, 320, 610], "3": [310, 520, 550, 630], "4": [620, 540, 500, 340], "5": [320, 410, 560, 640]}},
        "Day 3": {"Tariff": [0.24, 0.21, 0.26, 0.30], "Carbon": [500, 455, 505, 550], "Baseline load": [5.4, 5.0, 4.9, 6.4], "Spatial carbon": {"1": [540, 500, 320, 600], "2": [320, 510, 540, 600], "3": [560, 330, 520, 610], "4": [620, 560, 500, 330], "5": [330, 420, 550, 640]}},
        "Day 4": {"Tariff": [0.19, 0.24, 0.28, 0.22], "Carbon": [495, 470, 500, 535], "Baseline load": [5.0, 5.1, 5.0, 6.7], "Spatial carbon": {"1": [320, 520, 560, 600], "2": [550, 330, 520, 580], "3": [600, 540, 500, 320], "4": [560, 500, 330, 540], "5": [500, 340, 560, 630]}},
        "Day 5": {"Tariff": [0.23, 0.20, 0.27, 0.31], "Carbon": [500, 450, 505, 545], "Baseline load": [5.2, 5.3, 5.0, 6.6], "Spatial carbon": {"1": [510, 330, 560, 600], "2": [560, 500, 320, 590], "3": [320, 520, 540, 620], "4": [630, 560, 510, 340], "5": [330, 420, 560, 630]}},
        "Day 6": {"Tariff": [0.26, 0.22, 0.25, 0.29], "Carbon": [505, 460, 495, 540], "Baseline load": [5.5, 5.2, 4.8, 6.5], "Spatial carbon": {"1": [540, 500, 320, 610], "2": [320, 510, 560, 620], "3": [560, 340, 520, 610], "4": [640, 560, 510, 330], "5": [520, 330, 540, 600]}},
        "Day 7": {"Tariff": [0.21, 0.23, 0.28, 0.26], "Carbon": [495, 460, 500, 530], "Baseline load": [5.1, 4.9, 4.8, 6.3], "Spatial carbon": {"1": [330, 520, 560, 610], "2": [540, 330, 520, 600], "3": [580, 540, 330, 620], "4": [630, 560, 500, 330], "5": [520, 330, 550, 600]}}
    }
}

AGENT_ID = 2
LOCATION = 2
BASE_DEMAND = [0.70, 1.00, 0.80, 0.50]
SLOT_MAX_SESSIONS = CONTEXT['slot_max_sessions']

def calculate_load_metric(day_data, base_demand):
    # Primary goal: prioritize low baseline load for headroom (Location 2 analyst)
    metrics = []
    baseline_loads = day_data['Baseline load']
    
    for i in range(4):
        load_proxy = baseline_loads[i]
        
        # Heuristic penalty for slots near minimum session count if load is low
        if baseline_loads[i] < CONTEXT['slot_min_sessions'][i] * 2.0:
             load_proxy += 10.0 

        # Penalty for slots severely restricted by max sessions (slot 2 has max 1)
        if SLOT_MAX_SESSIONS[i] == 1:
            load_proxy += 0.5 

        metrics.append(load_proxy)
        
    return metrics

def select_slot(day_data):
    load_metrics = calculate_load_metric(day_data, BASE_DEMAND)
    
    min_score = min(load_metrics)
    best_slots = [i for i, score in enumerate(load_metrics) if score == min_score]
    
    # Tie-breaker: Lowest Tariff (secondary goal, departing from parents who might use Carbon/Spatial Carbon)
    if len(best_slots) > 1:
        tariffs = day_data['Tariff']
        best_tariff = float('inf')
        final_choice = -1
        
        for slot_idx in best_slots:
            if tariffs[slot_idx] < best_tariff:
                best_tariff = tariffs[slot_idx]
                final_choice = slot_idx
        
        return final_choice
    
    return best_slots[0]


def generate_policy():
    policy = []
    # Process days in numerical order
    day_keys = sorted(CONTEXT['days'].keys(), key=lambda x: int(x.split(' ')[1]))
    
    for day_key in day_keys:
        day_data = CONTEXT['days'][day_key]
        chosen_slot = select_slot(day_data)
        policy.append(chosen_slot)
        
    return policy

if __name__ == "__main__":
    final_policy = generate_policy()
    
    output_filename = "local_policy_output.json"
    with open(output_filename, 'w') as f:
        json.dump(final_policy, f)