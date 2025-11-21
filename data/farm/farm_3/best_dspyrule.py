import json
import math

def calculate_interventions(features):
    output_features = []
    
    # Define crops prioritized for full conversion based on low inherent market price 
    # (Oats: 95, Barley: 120 USD/Tonne).
    LOW_PRICE_CROPS_FOR_CONVERSION = {'Oats', 'Barley'}
    LOW_YIELD_THRESHOLD = 1.0

    for feature in features:
        if feature['properties'].get('type') == 'ag_plot':
            props = feature['properties']
            plot_id = props['id']
            yield_val = props['yield']
            label = props['label']
            
            # Default Margin Intervention: Max margin intervention is the most common outcome for plots receiving services.
            margin_intervention = 1.0
            habitat_conversion = 0.0
            
            should_convert_habitat = False
            
            # Habitat Conversion Heuristic based on low opportunity cost or low market value:
            
            # Case A: Low productivity plots (Yield <= 1.0)
            if yield_val <= LOW_YIELD_THRESHOLD:
                should_convert_habitat = True
            
            # Case B: Plots of low market value crops (Oats/Barley) regardless of high yield
            elif label in LOW_PRICE_CROPS_FOR_CONVERSION:
                should_convert_habitat = True

            if should_convert_habitat:
                habitat_conversion = 1.0
            
            # Ensure values are capped between 0 and 1
            margin_intervention = max(0.0, min(1.0, margin_intervention))
            habitat_conversion = max(0.0, min(1.0, habitat_conversion))
            
            output_props = {
                'id': plot_id,
                'margin_intervention': float(margin_intervention),
                'habitat_conversion': float(habitat_conversion)
            }
            
            output_features.append({
                'type': 'Feature',
                'properties': output_props,
                'geometry': feature['geometry']
            })

    output_geojson = {
        'type': 'FeatureCollection',
        'features': output_features
    }
    
    return output_geojson

# Load input data
try:
    with open('input.geojson', 'r') as f:
        input_data = json.load(f)
except FileNotFoundError:
    # Provide an empty structure if file loading fails, though it should exist.
    input_data = {'features': []} 

# Calculate interventions
dspy_output = calculate_interventions(input_data.get('features', []))

# Write output
with open('dspy_output.geojson', 'w') as f:
    # Use default separators for compact output, consistent with typical GeoJSON requirements
    json.dump(dspy_output, f)