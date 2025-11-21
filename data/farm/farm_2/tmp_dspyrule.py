import json
import os

def calculate_interventions(features):
    # Crop Prices (USD/Tonne)
    prices = {
        'Soybeans': 370, 
        'Oats': 95, 
        'Corn': 190, 
        'Canola/rapeseed': 1100, 
        'Barley': 120, 
        'Spring wheat': 200
    }

    output_features = []

    for feature in features:
        props = feature['properties']
        if props.get('type') == 'ag_plot':
            
            yield_val = props['yield']
            label = props['label']
            
            price = prices.get(label, 0)
            revenue_density = yield_val * price
            
            margin_intervention = 0.0
            habitat_conversion = 0.0
            
            # --- Heuristic Logic derived from ICL ---
            
            # 1. Default to full margin intervention unless contradicted by high yield/specific crop profile
            margin_intervention = 1.0
            
            # 2. Habitat conversion (HC) is generally avoided if revenue density is high (>1000)
            # or if the opportunity cost is significant.
            
            if revenue_density < 300:
                # Very low revenue plots often get full conversion (HC=1.0)
                habitat_conversion = 1.0
            elif revenue_density >= 1000:
                # High revenue density plots avoid conversion
                habitat_conversion = 0.0
            else:
                # Moderate revenue (300 <= R < 1000)
                habitat_conversion = 0.0
                
                # Special case: High yielding Soybeans near the upper moderate revenue limit 
                # tend to receive fractional margin intervention (based on N2 ID 8, R~969)
                if label == 'Soybeans' and yield_val > 2.5:
                    margin_intervention = 0.35 

            # --- Apply and store results ---
            
            new_props = feature['properties'].copy()
            new_props['margin_intervention'] = min(1.0, max(0.0, margin_intervention))
            new_props['habitat_conversion'] = min(1.0, max(0.0, habitat_conversion))
            
            output_feature = {
                'type': 'Feature',
                'properties': new_props,
                'geometry': feature['geometry']
            }
            output_features.append(output_feature)
            
    return {"type": "FeatureCollection", "features": output_features}

# --- Main Execution ---
def main():
    try:
        # Load input.geojson
        with open('input.geojson', 'r') as f:
            input_data = json.load(f)
        
        # Calculate interventions
        output_data = calculate_interventions(input_data['features'])
        
        # Write output to dspy_output.geojson
        with open('dspy_output.geojson', 'w') as f:
            json.dump(output_data, f, indent=2)

    except FileNotFoundError:
        # Expected error if running outside the farm environment structure
        pass
    except Exception:
        # Silent exception handling as required
        pass

if __name__ == '__main__':
    main()