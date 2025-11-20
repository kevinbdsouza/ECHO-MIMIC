import json

def calculate_interventions_new_heuristic(input_geojson_file, output_geojson_file):
    """
    Calculates and assigns interventions (margin and habitat conversion) to agricultural plots
    based on a new heuristic approach considering profitability.

    This heuristic compares the potential revenue from an agricultural plot with the implementation costs
    of margin and habitat interventions to decide on the most suitable intervention.
    It considers a default crop (Soybeans) and its price, and the implementation costs for margin and habitat interventions.

    Heuristic logic:
    1. For each 'ag_plot', calculate the potential revenue: revenue = yield * soybean_price.
    2. Compare revenue with margin and habitat intervention implementation costs.
    3. If revenue > margin_implementation_cost and revenue > habitat_implementation_cost:
       Choose the intervention with higher profit (revenue - cost). In practice, since habitat_cost < margin_cost, habitat will be chosen if revenue is high enough for both. Let's prioritize margin in this case for less disruptive intervention. So, choose margin.
    4. If revenue > margin_implementation_cost and revenue <= habitat_implementation_cost: Choose margin intervention.
    5. If revenue <= margin_implementation_cost and revenue > habitat_implementation_cost: Choose habitat intervention.
    6. If revenue <= margin_implementation_cost and revenue <= habitat_implementation_cost: No intervention (set both to 0).
    7. For non-'ag_plot' features, set both interventions to 0.

    Args:
        input_geojson_file (str): Path to the input GeoJSON file.
        output_geojson_file (str): Path to the output GeoJSON file to save results.

    Returns:
        str: Path to the output GeoJSON file.
    """

    crop_prices = {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, 'Barley': 120, 'Spring wheat': 200}
    intervention_costs = {
        'margin': {'implementation': 400,  'maintenance': 60},
        'habitat': {'implementation': 300, 'maintenance': 70},
        'agriculture': {'maintenance': 100}
    }
    soybean_price = crop_prices['Soybeans']
    margin_implementation_cost = intervention_costs['margin']['implementation']
    habitat_implementation_cost = intervention_costs['habitat']['implementation']

    with open(input_geojson_file, 'r') as f:
        input_data = json.load(f)

    output_features = []
    for feature in input_data['features']:
        props = feature['properties']
        if props['type'] == 'ag_plot':
            yield_value = props['yield']
            revenue = yield_value * soybean_price

            if revenue > margin_implementation_cost:
                if revenue > habitat_implementation_cost:
                    # Choose margin if both are profitable or equally profitable
                    props['margin_intervention'] = 1.0
                    props['habitat_conversion'] = 0.0
                else:
                    props['margin_intervention'] = 1.0
                    props['habitat_conversion'] = 0.0
            else:
                if revenue > habitat_implementation_cost:
                    props['margin_intervention'] = 0.0
                    props['habitat_conversion'] = 1.0
                else:
                    props['margin_intervention'] = 0.0
                    props['habitat_conversion'] = 0.0

        else:
            props['margin_intervention'] = 0.0
            props['habitat_conversion'] = 0.0
        output_features.append(feature)

    output_geojson = {
        "type": "FeatureCollection",
        "features": output_features
    }

    with open(output_geojson_file, 'w') as f:
        json.dump(output_geojson, f, indent=2)

    return output_geojson_file

# Use the function to process input.geojson and save to output.geojson
input_file = 'input.geojson'
output_file = 'output.geojson'
output_filepath = calculate_interventions_new_heuristic(input_file, output_file)
print(f"Output GeoJSON with interventions saved to: {output_filepath}")