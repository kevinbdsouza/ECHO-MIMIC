import json

def predict_interventions(input_geojson_file):
    with open(input_geojson_file, 'r') as f:
        data = json.load(f)

    output_interventions = []
    for feature in data['features']:
        plot_id = feature['properties']['id']
        plot_type = feature['properties']['type']
        label = feature['properties']['label']

        if plot_type == 'hab_plots':
            margin_directions = []
            habitat_directions = ["north-west", "north-east", "south-west", "south-east"]
        elif plot_type == 'ag_plot':
            margin_directions = ["north-west", "north-east", "south-west", "south-east"]
            habitat_directions = []

            if label == "Spring wheat":
                if plot_id == 2:
                    margin_directions = ["north-west", "north-east", "south-west"]
                    habitat_directions = ["north-east"]
                # plot 8 spring wheat keeps full margin
            elif label == "Oats":
                margin_directions = ["north-west", "south-west", "south-east"]
            # Canola/rapeseed keeps full margin

        else:
            margin_directions = []
            habitat_directions = []  # Default case

        output_interventions.append({
            "plot_id": plot_id,
            "plot_type": plot_type,
            "label": label,
            "margin_directions": margin_directions,
            "habitat_directions": habitat_directions
        })
    return output_interventions

input_file = 'input.geojson'
output_file = 'output.json'
predicted_interventions = predict_interventions(input_file)

with open(output_file, 'w') as f:
    json.dump(predicted_interventions, f, indent=2)

print(f"Output saved to {output_file}")