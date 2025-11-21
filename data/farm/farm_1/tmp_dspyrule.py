import json
import math
import os

def run_intervention_planning():
    """
    Loads input.geojson, determines intervention values for agricultural plots 
    based on Value Potential (Yield * Price), and saves the output to dspy_output.geojson.
    """
    
    # 1. Load input.geojson
    try:
        with open('input.geojson', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Silent exit if file is not found, adhering to constraints (though usually guaranteed present)
        return

    # Define parameters (based on instructions/neighbor context)
    CROP_PRICES = {
        'Soybeans': 370, 
        'Oats': 95, 
        'Corn': 190, 
        'Canola/rapeseed': 1100, 
        'Barley': 120, 
        'Spring wheat': 200
    }
    
    # Heuristic Threshold for Value Potential (V = Yield * Price).
    # Based on ICL analysis, 350 separates most high-revenue plots (HC=0) 
    # from low-revenue plots (HC=1).
    V_THRESHOLD = 350.0

    output_features = []
    
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        
        if props.get('type') == 'ag_plot':
            plot_id = props.get('id')
            label = props.get('label')
            ag_yield = props.get('yield', 0.0)

            price = CROP_PRICES.get(label, 0.0)
            V = ag_yield * price # Value Potential approximation
            
            # --- Intervention Logic ---
            
            # 1. Margin Intervention: Set to 1.0 (dominant strategy observed in ICL)
            margin_intervention = 1.0
            
            # 2. Habitat Conversion: Based on Value Potential V
            if V < V_THRESHOLD:
                # Low revenue potential (V < 350) -> convert fully
                habitat_conversion = 1.0
            else:
                # Moderate/High revenue potential -> avoid conversion
                habitat_conversion = 0.0
                
            
            new_properties = {
                "id": plot_id,
                "margin_intervention": float(margin_intervention),
                "habitat_conversion": float(habitat_conversion)
            }
            
            # Create the output feature, copying geometry
            output_feature = {
                "type": "Feature",
                "properties": new_properties,
                "geometry": feature.get('geometry')
            }
            output_features.append(output_feature)

    # 3. Write output to dspy_output.geojson
    output_geojson = {
        "type": "FeatureCollection",
        "features": output_features
    }
    
    with open('dspy_output.geojson', 'w') as f:
        json.dump(output_geojson, f, indent=2)

if __name__ == "__main__":
    run_intervention_planning()