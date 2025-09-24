```python
import json

def predict_interventions(features):
    # Heuristics based on neighbor examples:
    # - Higher yield crops tend to have lower habitat conversion.
    # - Lower yield crops tend to have higher habitat conversion.
    # - Margin intervention is generally high across all crops.

    crop_prices = {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, 'Barley': 120, 'Spring wheat': 200}
    costs = {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}

    interventions = []
    for feature in features:
        if feature['properties']['type'] == 'ag_plot':
            yield_val = feature['properties']['yield']
            crop_type = feature['properties']['label']
            margin_intervention = 1.0  # Generally high margin intervention

            # Habitat conversion based on yield:
            if yield_val > 2.0:
                habitat_conversion = 0.0
            elif yield_val < 1.0:
                habitat_conversion = 1.0
            else:
                habitat_conversion = 0.5

            # Adjust based on crop price (higher price, lower habitat conversion):
            habitat_conversion *= (1 - (crop_prices[crop_type] / 1200))

            interventions.append({
                'type': 'Feature',
                'properties': {
                    'id': feature['properties']['id'],
                    'label': crop_type,
                    'type': 'ag_plot',
                    'yield': yield_val,
                    'margin_intervention': max(0.0, min(1.0, margin_intervention)),
                    'habitat_conversion': max(0.0, min(1.0, habitat_conversion))
                },
                'geometry': feature['geometry']
            })
    return interventions


with open('input.geojson', 'r') as f:
    data = json.load(f)

output_features = predict_interventions(data['features'])

output_data = {
    'type': 'FeatureCollection',
    'features': output_features
}

with open('dspy_output.geojson', 'w') as f:
    json.dump(output_data, f, indent=2)

```