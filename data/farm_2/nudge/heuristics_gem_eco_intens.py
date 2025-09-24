import json

def calculate_interventions(input_file="input.geojson", output_file="output.geojson"):
    """
    Calculates and assigns margin and habitat interventions for agricultural plots based on heuristics.

    Args:
        input_file (str): Path to the input GeoJSON file.
        output_file (str): Path to save the output GeoJSON file.
    """

    crop_prices = {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, 'Barley': 120, 'Spring wheat': 200}
    costs = {'margin': {'implementation': 400, 'maintenance': 60}, 'habitat': {'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}

    with open(input_file, 'r') as f:
        data = json.load(f)

    for feature in data['features']:
        if feature['properties']['type'] == 'ag_plot':
            # Initialize intervention values
            feature['properties']['margin_intervention'] = 0.0
            feature['properties']['habitat_conversion'] = 0.0

            # Heuristics based on yield and crop type

            yield_value = feature['properties']['yield']
            label = feature['properties']['label']

            # High Yield Threshold
            high_yield_threshold = 2.0

            if yield_value > high_yield_threshold:
                feature['properties']['margin_intervention'] = 1.0 # Apply full margin intervention
            elif label == "Oats":
                  feature['properties']['margin_intervention'] = 0.1
            elif yield_value > 1:
                feature['properties']['margin_intervention'] = 0.75
            else:
                feature['properties']['margin_intervention'] = 0.0 # No intervention for low yield

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    calculate_interventions()