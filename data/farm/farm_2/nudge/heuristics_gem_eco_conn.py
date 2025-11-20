import json

def predict_interventions(input_geojson_file, output_geojson_file):
    with open(input_geojson_file, 'r') as f:
        input_data = json.load(f)

    output_features = []
    for feature in input_data['features']:
        plot_properties = feature['properties']
        plot_type = plot_properties['type']
        label = plot_properties['label']
        plot_id = plot_properties['id']

        margin_directions = []
        habitat_directions = []

        if plot_type == 'hab_plots':
            habitat_directions = ["north-west", "north-east", "south-west", "south-east"]
        elif plot_type == 'ag_plot':
            if label == 'Soybeans':
                margin_directions = ["north-west", "north-east", "south-west", "south-east"]
            else:
                margin_directions = ["north-east", "south-east"]

        output_features.append({
            "plot_id": plot_id,
            "plot_type": plot_type,
            "label": label,
            "margin_directions": margin_directions,
            "habitat_directions": habitat_directions
        })

    output_json = output_features

    with open(output_geojson_file, 'w') as f:
        json.dump(output_json, f, indent=2)

    return output_json

# Assuming input.geojson is in the same directory
input_file = 'input.geojson'
output_file = 'output.json'
output_predictions = predict_interventions(input_file, output_file)
print(output_predictions)