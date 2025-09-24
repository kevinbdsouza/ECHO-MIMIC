import json

# Crop prices in USD/Tonne
crop_prices = {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, 'Barley': 120, 'Spring wheat': 200}
# Costs (implementation costs one time and in USD/ha, and maintenance costs in USD/ha/year)
costs = {'margin': {'implementation': 400, 'maintenance': 60},
         'habitat': {'implementation': 300, 'maintenance': 70},
         'agriculture': {'maintenance': 100}}

def calculate_interventions(input_file="input.geojson", output_file="output.geojson"):
    """
    Combined Heuristic: Integrates elements from Parent 1 and Parent 2, focusing on cost-benefit analysis,
    dynamic crop-specific adjustments, landscape context, and neighbor influences to determine
    margin intervention and habitat conversion values for each agricultural plot.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    # Calculate the median yield for ag_plots
    ag_plot_yields = [
        feature["properties"]["yield"]
        for feature in data["features"]
        if feature["properties"]["type"] == "ag_plot"
    ]
    median_yield = sorted(ag_plot_yields)[len(ag_plot_yields) // 2] if ag_plot_yields else 0

    # Landscape context: Calculate habitat ratio
    ag_plot_count = sum(1 for feature in data["features"] if feature["properties"]["type"] == "ag_plot")
    hab_plot_count = sum(1 for feature in data["features"] if feature["properties"]["type"] == "hab_plot")
    total_plot_count = ag_plot_count + hab_plot_count
    habitat_ratio = hab_plot_count / total_plot_count if total_plot_count > 0 else 0.0

    output_features = []
    for i, feature in enumerate(data["features"]):
        if feature["properties"]["type"] == "ag_plot":
            properties = feature['properties']
            crop_label = properties.get('label', 'Corn')  # Default to Corn
            yield_value = properties.get("yield", 1.0)  # Default to 1.0
            plot_area = properties.get('area', 1.0)  # Default to 1.0

            crop_price = crop_prices.get(crop_label, 100)  # Default to 100
            intervention_score = yield_value * crop_price

            # Incorporate costs (using agriculture maintenance cost)
            intervention_score -= costs['agriculture']['maintenance'] * 0.1 # crude cost estimation

            # Cost-benefit analysis: approximate cost of margin intervention over 5 years
            margin_cost_5yr = (costs['margin']['implementation'] + (5 * costs['margin']['maintenance'])) * (plot_area/10000) #Scale to ha
            margin_cost_10yr = (costs['margin']['implementation'] + (10 * costs['margin']['maintenance'])) * (plot_area/10000) #Scale to ha

            margin_intervention = 0.0
            habitat_conversion = 0.0

            # Enhanced Margin Intervention Heuristics
            if yield_value >= median_yield:
                cost_benefit_ratio = intervention_score / margin_cost_5yr if margin_cost_5yr > 0 else 0
                if intervention_score > margin_cost_5yr * 1.1: # Slightly increased threshold for full margin
                    margin_intervention = 1.0
                elif cost_benefit_ratio > 0.8: # Adjusted thresholds for smoother transition
                     margin_intervention = 0.8
                elif cost_benefit_ratio > 0.6:
                    margin_intervention = 0.6 + (cost_benefit_ratio - 0.6) * 0.25 # Scaled between 0.6 and 0.8
                elif cost_benefit_ratio > 0.4:
                    margin_intervention = 0.4 + (cost_benefit_ratio - 0.4) * 0.2 # Scaled between 0.4 and 0.6
                else:
                    margin_intervention = 0.3  # Minimum intervention, slightly reduced

                # Area scaling factor - using consistent scaling
                area_scaling_factor = 1 + (plot_area / 15000) * 0.05
                margin_intervention *= area_scaling_factor
                margin_intervention = min(margin_intervention, 1.0)

                # Yield difference scaling - non-linear scaling
                margin_intervention += min(0.15, ((yield_value - median_yield) / median_yield)**0.5 ) # Reduced max contribution
                margin_intervention = min(margin_intervention, 1.0)

            else:
                margin_intervention = 0.0 # Conservative approach for below median yield


            # Refined Habitat Conversion Heuristics

            # Dynamic Crop-specific adjustments
            if crop_label in ['Oats', 'Soybeans']:
                if yield_value < 0.95: # Adjusted yield threshold
                    if crop_price < 420: # Adjusted price threshold
                        habitat_conversion = min(1.0, 1.6/(plot_area/10000 + 0.2))  # Slightly increased base conversion
                    else:
                        margin_intervention = max(margin_intervention, 0.15) # Slight margin even with habitat conversion
            elif crop_label == 'Corn':
                 margin_intervention = min(1.0, margin_intervention * 0.7) # Further reduced corn margin
            elif crop_label == 'Spring wheat':
                if yield_value < 0.55: # Adjusted yield threshold
                    habitat_conversion = max(habitat_conversion, 0.4)
                    margin_intervention = max(margin_intervention, 0.8)
                    if crop_price > 155: # Adjusted price threshold
                        margin_intervention = max(margin_intervention, 0.93) # Increased margin for valuable wheat
            elif crop_label == "Canola/rapeseed":
                if yield_value < 1.45: # Adjusted yield threshold
                     margin_intervention = max(margin_intervention, 0.86)  # Increased margin for low-yield canola


            # Stronger prevention for high crop value
            if crop_price > 850: # Adjusted price threshold
                habitat_conversion = 0.0
                margin_intervention = min(1.0, margin_intervention * 1.35) # Further increased margin for high value


            # More direct yield based conversion - adjusted thresholds
            if intervention_score < margin_cost_10yr * 0.9 and yield_value < 0.68: # Adjusted thresholds
                 habitat_conversion = max(habitat_conversion, 0.72) # Base habitat conversion
            elif yield_value < 0.48: # Adjusted yield threshold
                habitat_conversion = max(habitat_conversion, 0.92) # Higher conversion for very low yield
            elif yield_value < 0.73: # Adjusted yield threshold
                habitat_conversion = max(habitat_conversion, 0.53) # Ensure some habitat conversion


            # Adjusted Habitat conversion calculation
            yield_maintenance_ratio = yield_value / costs['agriculture']['maintenance'] if costs['agriculture']['maintenance'] > 0 else 0
            habitat_conversion += (crop_price / 470) * (yield_maintenance_ratio / 8.75)  # Adjusted scaling

            habitat_conversion -= (costs['habitat']['implementation'] / 940) * habitat_conversion # Adjusted penalty

            habitat_conversion_threshold = 0.715 + (plot_area / 8600) - (crop_price / 1860)  # Adjusted threshold parameters
            if yield_value < habitat_conversion_threshold:
                habitat_conversion = max(habitat_conversion, 0.95) # Slightly reduced max habitat conversion
            else:
                habitat_conversion = min(habitat_conversion, 0.0)

            # Neighbor influence: Check previous and next plots in feature list - REFINED from Heuristic 4
            neighbor_influence = 0.0
            neighbor_threshold_habitat = 0.5
            neighbor_influence_factor = 0.2 # Increased influence factor
            neighbor_count = 0  # Count of neighbors meeting the threshold

            if i > 0:
                neighbor_feature = data["features"][i-1]
                if neighbor_feature["properties"]["type"] == "ag_plot" and neighbor_feature["properties"].get("habitat_conversion", 0) >= neighbor_threshold_habitat:
                   neighbor_count += 1
                # Check neighbor of neighbor (before)
                if i > 1:
                    neighbor_neighbor_feature = data["features"][i-2]
                    if neighbor_neighbor_feature["properties"]["type"] == "hab_plot": #Only consider habitat plots
                         neighbor_count += 1


            if i < len(data["features"]) - 1:
                neighbor_feature = data["features"][i+1]
                if neighbor_feature["properties"]["type"] == "ag_plot" and neighbor_feature["properties"].get("habitat_conversion", 0) >= neighbor_threshold_habitat:
                    neighbor_count += 1
                # Check neighbor of neighbor (after)
                if i < len(data["features"]) - 2:
                     neighbor_neighbor_feature = data["features"][i+2]
                     if neighbor_neighbor_feature["properties"]["type"] == "hab_plot": #Only consider habitat plots
                         neighbor_count += 1

            neighbor_influence = neighbor_influence_factor * neighbor_count
            habitat_conversion += neighbor_influence


            # Enhanced Landscape context influence - Non-linear scaling for habitat_conversion
            habitat_conversion *= (1.6 - habitat_ratio)**1.1 # Non-linear scaling, stronger at low habitat ratios
            margin_intervention *= (1 + 0.4 * (1 - habitat_ratio)) # Slightly reduced landscape scaling for margin


            habitat_conversion = min(habitat_conversion, 1.0)
            habitat_conversion = max(habitat_conversion, 0.0)
            margin_intervention = min(margin_intervention, 1.0)
            margin_intervention = max(margin_intervention, 0.0)


            properties['margin_intervention'] = margin_intervention
            properties['habitat_conversion'] = habitat_conversion
            output_features.append(feature)

        else:
            feature["properties"]["margin_intervention"] = 0.0
            feature["properties"]["habitat_conversion"] = 0.0
            output_features.append(feature)


    output_data = {
        "type": "FeatureCollection",
        "features": output_features
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

calculate_interventions()