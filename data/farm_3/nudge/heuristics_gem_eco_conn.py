import json

def process_plots():
    with open('input.geojson', 'r') as f:
        input_data = json.load(f)

    output_features = []
    for feature in input_data['features']:
        plot_properties = feature['properties']
        plot_type = plot_properties['type']
        plot_label = plot_properties['label']
        plot_id = plot_properties['id']

        margin_directions = []
        habitat_directions = []

        if plot_type == 'ag_plot':
            if plot_label == "Corn":
                margin_directions = ["north-west", "north-east", "south-west", "south-east"]
            elif plot_label == "Soybeans" and plot_id == 3:
                margin_directions = ["south-west", "south-east"]
            elif plot_label == "Spring wheat":
                margin_directions = ["north-east", "south-west"]
                habitat_directions = ["north-west", "north-east", "south-west", "south-east"]
            elif plot_label == "Barley":
                margin_directions = ["north-west", "south-west", "south-east"]
                habitat_directions = ["north-west", "north-east", "south-west", "south-east"]
            elif plot_label == "Soybeans" and plot_id == 6:
                margin_directions = ["north-west", "north-east", "south-east"]
            elif plot_label == "Oats":
                margin_directions = ["north-west", "north-east", "south-east"]
                habitat_directions = ["north-west", "north-east", "south-west", "south-east"]

        output_feature = {
            "id": plot_id,
            "type": plot_type,
            "label": plot_label,
            "margin_directions": margin_directions,
            "habitat_directions": habitat_directions
        }
        if plot_type == 'ag_plot':
            output_features.append(output_feature)

    with open('output.json', 'w') as f:
        json.dump(output_features, f, indent=2)

if __name__ == '__main__':
    process_plots()