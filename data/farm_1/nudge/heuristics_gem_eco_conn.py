import json
import math
import os
import numpy
import pandas
import scipy
import shapely
import geopandas


def calculate_directions(input_file="input.geojson"):
    """
    This is the main function to calculate heuristics for directions
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    output_features = []

    for feature in data['features']:
        if feature['properties']['type'] == 'ag_plot':
            plot_label = feature['properties']['label']
            plot_yield = feature['properties']['yield']

            margin_directions = []
            habitat_directions = []

            if plot_label == "Spring wheat":
                if plot_yield <= 1.0:
                    margin_directions = ["north-east", "south-west", "south-east"]
                    habitat_directions = ["north-east", "south-east"]
                elif 1.0 < plot_yield < 2.0:
                    margin_directions = ["south-east"]
                elif plot_yield >= 2.0:
                    margin_directions = ["north-west", "north-east", "south-west", "south-east"]
            elif plot_label == "Oats":
                margin_directions = ["north-west", "north-east", "south-west", "south-east"]
                habitat_directions = ["south-east"]
            elif plot_label == "Corn":
                margin_directions = ["north-east", "south-east"]
            elif plot_label == "Soybeans":
                margin_directions = ["north-west", "north-east", "south-west", "south-east"]
            elif plot_label == "Canola/rapeseed":
                margin_directions = ["north-west", "north-east", "south-west", "south-east"]
            else:
                margin_directions = []
                habitat_directions = []

            feature['properties']['margin_directions'] = margin_directions
            feature['properties']['habitat_directions'] = habitat_directions
            output_features.append(feature['properties'])
        else:
            feature['properties']['margin_directions'] = []
            feature['properties']['habitat_directions'] = []
            output_features.append(feature['properties'])

    with open('output.json', 'w') as f:
        json.dump(output_features, f, indent=2)


#Call the calculate_directions() function here to run through all the steps
calculate_directions()