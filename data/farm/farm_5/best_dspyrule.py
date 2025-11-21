import json
import math
import os

# Define the intervention logic based on observed patterns relating yield and crop label
def calculate_interventions(features):
    output_features = []
    
    # Heuristic Parameters derived from ICL examples for gradient crops (Spring Wheat and Soybeans)
    # These parameters define the yield range over which conversion scales linearly from 1.0 (low yield) to 0.0 (high yield).
    # Using 2.80 as maximum viable yield for Spring Wheat (based on E3 ID 1)
    SW_MAX_YIELD = 2.80 
    SW_MIN_YIELD = 0.50
    SW_YIELD_RANGE = SW_MAX_YIELD - SW_MIN_YIELD
    
    # Using 2.65 as maximum viable yield for Soybeans (slightly above E3 ID 8, E1 ID 5)
    SOY_MAX_YIELD = 2.65
    SOY_MIN_YIELD = 0.50
    SOY_YIELD_RANGE = SOY_MAX_YIELD - SOY_MIN_YIELD

    for feature in features:
        if feature['properties']['type'] == 'ag_plot':
            props = feature['properties']
            label = props['label']
            plot_id = props['id']
            y = props['yield']
            
            # Margin Intervention (M): Typically 1.0 for plots receiving intervention
            margin_intervention = 1.0
            habitat_conversion = 0.0
            
            # --- Habitat Conversion (H) Logic ---
            
            # 1. High Value Crops (Canola/rapeseed) -> Protect from conversion (H=0.0)
            if label == 'Canola/rapeseed':
                habitat_conversion = 0.0
            
            # 2. Low Value Crops (Oats, Barley) -> Full conversion (H=1.0)
            elif label == 'Oats' or label == 'Barley':
                habitat_conversion = 1.0
            
            # 3. Spring Wheat -> Gradient based on yield deficiency
            elif label == 'Spring wheat':
                if SW_YIELD_RANGE > 1e-6:
                    deficiency = SW_MAX_YIELD - y
                    # Normalize deficiency to [0, 1] range based on observed yield range
                    h_val = deficiency / SW_YIELD_RANGE
                    habitat_conversion = max(0.0, min(1.0, h_val))
                else:
                    habitat_conversion = 0.0
            
            # 4. Soybeans -> Gradient based on yield deficiency
            elif label == 'Soybeans':
                if SOY_YIELD_RANGE > 1e-6:
                    deficiency = SOY_MAX_YIELD - y
                    h_val = deficiency / SOY_YIELD_RANGE
                    habitat_conversion = max(0.0, min(1.0, h_val))
                else:
                    habitat_conversion = 0.0
            
            # 5. Catch-all: default to minimum conversion
            else:
                habitat_conversion = 0.0
            

            # Ensure final values are floats and clipped to [0, 1]
            margin_intervention = float(max(0.0, min(1.0, margin_intervention)))
            habitat_conversion = float(max(0.0, min(1.0, habitat_conversion)))
            
            # Create output feature
            out_feature = {
                "type": "Feature",
                "properties": {
                    "id": plot_id,
                    "margin_intervention": margin_intervention,
                    "habitat_conversion": habitat_conversion
                },
                "geometry": feature['geometry']
            }
            output_features.append(out_feature)
            
    return {
        "type": "FeatureCollection",
        "features": output_features
    }

if __name__ == "__main__":
    input_filename = 'input.geojson'
    output_filename = 'dspy_output.geojson' 
    
    # 1. Load input data
    with open(input_filename, 'r') as f:
        input_data = json.load(f)
    
    # 2. Calculate interventions
    output_data = calculate_interventions(input_data['features'])
    
    # 3. Write output data
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)