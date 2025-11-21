import json

def get_interventions(features):
    # Crop prices USD/Tonne
    PRICES = {
        'Soybeans': 370,
        'Oats': 95,
        'Corn': 190,
        'Canola/rapeseed': 1100,
        'Barley': 120,
        'Spring wheat': 200
    }

    output_features = []

    # Heuristic thresholds based on Economic Value (V = Price * Yield) derived from neighbor examples:
    # 1. V > 500 implies high value, warrants full margin intervention (1.0).
    # 2. V <= 150 for Spring Wheat/Barley implies extremely low revenue, warrants full conversion (1.0 H, 1.0 M), mirroring N3 specific behavior.
    # 3. Intermediate values or low yield high-price crops get partial margin intervention (0.5).
    
    V_HIGH_MARGIN = 500.0
    V_LOW_CONVERSION = 150.0

    for feature in features:
        if feature['properties']['type'] == 'ag_plot':
            properties = feature['properties']
            plot_id = properties['id']
            label = properties['label']
            yield_val = properties['yield']
            geometry = feature['geometry']
            
            price = PRICES.get(label, 0.0)
            V = price * yield_val

            margin_intervention = 0.0
            habitat_conversion = 0.0

            # Rule 1: High Value plots get full margin intervention
            if V > V_HIGH_MARGIN:
                margin_intervention = 1.0
                habitat_conversion = 0.0
            
            # Rule 2: Low Value plots treatment
            else:
                # 2.1 Critically low value crops (SW, Barley) eligible for full conversion
                if label in ["Spring wheat", "Barley"] and V <= V_LOW_CONVERSION:
                    margin_intervention = 1.0
                    habitat_conversion = 1.0
                
                # 2.2 Soybeans/Canola (Higher price crops) retain moderate margin intervention even if yield is low
                elif label in ["Soybeans", "Canola/rapeseed"]:
                    margin_intervention = 0.5
                    habitat_conversion = 0.0
                    
                # 2.3 Mid-to-low value cereals/grains (Oats, Corn, intermediate SW/Barley)
                elif label in ["Spring wheat", "Barley", "Oats", "Corn"]:
                    if V > V_LOW_CONVERSION: # 150 < V <= 500
                        margin_intervention = 0.5
                        habitat_conversion = 0.0
                    else: 
                        # Very low value, but not SW/Barley eligible for conversion (e.g., low yield Oats/Corn)
                        margin_intervention = 0.0
                        habitat_conversion = 0.0
                
                # Default case (should not occur with typical crops)
                else:
                    margin_intervention = 0.0
                    habitat_conversion = 0.0
            
            # Construct output feature, ensuring floats are used
            output_feature = {
                "type": "Feature",
                "properties": {
                    "id": plot_id,
                    "margin_intervention": float(margin_intervention),
                    "habitat_conversion": float(habitat_conversion)
                },
                "geometry": geometry
            }
            output_features.append(output_feature)

    output_collection = {
        "type": "FeatureCollection",
        "features": output_features
    }
    return output_collection

if __name__ == "__main__":
    with open('input.geojson', 'r') as f:
        input_data = json.load(f)

    output_data = get_interventions(input_data['features'])

    with open('dspy_output.geojson', 'w') as f:
        json.dump(output_data, f, indent=2)