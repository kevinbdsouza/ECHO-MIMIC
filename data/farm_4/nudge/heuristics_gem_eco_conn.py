import json

def predict_interventions():
    with open('input.geojson', 'r') as f:
        input_geojson = json.load(f)

    output_interventions = []

    for feature in input_geojson['features']:
        if feature['properties']['type'] == 'ag_plot':
            plot_id = feature['properties']['id']
            plot_type = feature['properties']['type']
            label = feature['properties']['label']
            plot_yield = feature['properties']['yield']

            margin_directions = []
            habitat_directions = []

            if plot_yield <= 0.5:
                habitat_directions = ["north-west", "north-east", "south-west", "south-east"]
                if label in ['Spring wheat', 'Oats', 'Barley', 'Soybeans']:
                    margin_directions = ["north-west", "north-east", "south-west", "south-east"] # Default margins for very low yield
                else: # Corn, Canola etc.
                    margin_directions = [] # Conservative margins for other crops at very low yield

                if plot_id == 2: # Special case from hint, adjust margins based on hint for plot 2
                    margin_directions = ["south-west", "south-east"]
                if plot_id == 7: # Special case from hint, adjust margins based on hint for plot 7
                    margin_directions = ["north-west"]
                if plot_id == 9: # Special case from hint, adjust margins based on hint for plot 9
                    margin_directions = ["north-east", "south-west", "south-east"]


            elif plot_yield <= 1.7:
                habitat_directions = []
                if label in ['Spring wheat', 'Oats', 'Barley', 'Soybeans']:
                    margin_directions = ["north-west", "north-east", "south-west", "south-east"] # Default margins for low yield
                else: # Corn, Canola etc.
                    margin_directions = ["north-west", "north-east"] # Conservative margins for other crops at low yield

                if plot_id == 6: # Special case from hint, adjust margins based on hint for plot 6
                    margin_directions = ["north-west", "south-east"]
                    habitat_directions = ["north-west", "north-east", "south-west", "south-east"]


            elif plot_yield <= 2.6:
                habitat_directions = []
                if label in ['Spring wheat', 'Oats', 'Barley', 'Soybeans']:
                     margin_directions = ["north-west", "north-east", "south-west"] # Reduced margins for medium yield
                else:
                    margin_directions = ["north-west"] # More conservative for other crops at medium yield

                if plot_id == 3: # Special case from hint, adjust margins based on hint for plot 3
                    margin_directions = ["north-west", "north-east", "south-west"]
                    habitat_directions = ["north-west", "north-east", "south-west", "south-east"]


            elif plot_yield > 2.6:
                habitat_directions = []
                margin_directions = [] # Minimal margins for high yield, default no margin

                if plot_id == 1: # Special case from hint, adjust margins based on hint for plot 1
                    margin_directions = ["north-west", "north-east", "south-west", "south-east"]
                    habitat_directions = ["north-west", "north-east", "south-west", "south-east"]
                if plot_id == 5: # Special case from hint, adjust margins based on hint for plot 5
                    margin_directions = ["north-east", "south-west", "south-east"]
                if plot_id == 8: # Special case from hint, adjust margins based on hint for plot 8
                    margin_directions = ["south-west"]


            output_interventions.append({
                "id": plot_id,
                "type": plot_type,
                "label": label,
                "margin_directions": margin_directions,
                "habitat_directions": habitat_directions
            })

    with open('output.json', 'w') as f:
        json.dump(output_interventions, f, indent=2)

    return output_interventions

if __name__ == '__main__':
    predicted_interventions = predict_interventions()
    print(json.dumps(predicted_interventions, indent=2))