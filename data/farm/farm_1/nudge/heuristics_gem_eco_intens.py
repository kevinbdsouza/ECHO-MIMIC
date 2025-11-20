import json
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

def calculate_polygon_area(coords):
    """Calculates polygon area using the shoelace formula."""
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def calculate_distance(coord1, coord2):
    """Calculates Euclidean distance between two coordinate pairs."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def is_within_bounds(point, vertices):
    """Check if a point is within the polygon defined by the vertices."""
    polygon = Delaunay(vertices)
    return polygon.find_simplex(point) >= 0

def enhanced_integrated_heuristic():
    """
    Integrates landscape connectivity, climate impact, socio-economic factors, NPV,
    slope, habitat adjacency, scarcity, and resource synergy for agricultural interventions.
    """
    try:
        with open('input.geojson', 'r') as f:
            input_geojson = json.load(f)
    except FileNotFoundError:
        print("Error: input.geojson not found.")
        return "Error: input.geojson not found."
    except json.JSONDecodeError:
        print("Error: input.geojson is not valid JSON.")
        return "Error: input.geojson is not valid JSON."

    ag_plots = [feature for feature in input_geojson['features'] if feature['properties'].get('type') == 'ag_plot']
    hab_plots = [feature for feature in input_geojson['features'] if feature['properties'].get('type') == 'hab_plot']
    water_plots = [feature for feature in input_geojson['features'] if feature['properties'].get('type') == 'water_plot']
    other_plots = [feature for feature in input_geojson['features'] if feature['properties']['type'] not in ['ag_plot', 'hab_plot', 'water_plot']]

    if not ag_plots:
        print("Warning: No agricultural plots found.")
        output_features_empty = []
        for feature in input_geojson['features']:
            plot_id = feature['properties']['id']
            output_features_empty.append({
                "type": "Feature",
                "properties": {
                    "id": plot_id,
                    "margin_intervention": 0,
                    "habitat_conversion": 0
                },
                "geometry": feature.get('geometry')
            })
        output_geojson_empty = {
            "type": "FeatureCollection",
            "features": output_features_empty
        }
        with open('output.geojson', 'w') as f:
            json.dump(output_geojson_empty, f, indent=2)
        return "No agricultural plots found, output.geojson created with default interventions."

    ag_coords = []
    ag_yields = []
    ag_areas = []
    ag_plot_features = []
    plot_areas = []
    total_ag_yield = 0
    total_ag_area = 0

    valid_ag_plots_found = False
    for feature in ag_plots:
        try:
            coords = feature['geometry']['coordinates'][0]
            if isinstance(coords, list) and len(coords) >= 3 and all(isinstance(coord, list) and len(coord) == 2 for coord in coords):
                centroid_coords = np.mean(coords, axis=0).tolist()
                ag_coords.append(centroid_coords)
                ag_yields.append(feature['properties']['yield'])
                plot_area = calculate_polygon_area(coords)
                ag_areas.append(plot_area)
                ag_plot_features.append(feature)
                total_ag_yield += feature['properties']['yield']
                total_ag_area += plot_area
                plot_areas.append(plot_area)
                valid_ag_plots_found = True
            else:
                print(f"Warning: Skipping plot with invalid coordinates: {feature['properties']['id']}")
        except (KeyError, TypeError) as e:
            print(f"Warning: Skipping plot due to error: {e}, Plot ID: {feature.get('properties', {}).get('id', 'Unknown')}")
            continue

    if not valid_ag_plots_found:
        print("Warning: No valid agricultural plots found after processing properties.")
        output_features_no_valid_ag = []
        for feature in input_geojson['features']:
            plot_id = feature['properties']['id']
            output_features_no_valid_ag.append({
                "type": "Feature",
                "properties": {
                    "id": plot_id,
                    "margin_intervention": 0,
                    "habitat_conversion": 0
                },
                "geometry": feature.get('geometry')
            })
        output_geojson_no_valid_ag = {
            "type": "FeatureCollection",
            "features": output_features_no_valid_ag
        }
        with open('output.geojson', 'w') as f:
            json.dump(output_geojson_no_valid_ag, f, indent=2)
        return "No valid agricultural plots found after processing, output.geojson created with default interventions."

    ag_coords = np.array(ag_coords)
    ag_yields = np.array(ag_yields)
    ag_areas = np.array(ag_areas)

    mean_yield_density = total_ag_yield / total_ag_area if total_ag_area > 0 else 0
    average_plot_area = sum(plot_areas) / len(plot_areas) if plot_areas else 0
    num_hab_plots = len(hab_plots)
    decay_constant = 0.05 + (0.1 / (1 + num_hab_plots)) # Adaptive decay constant

    hab_plot_data = []
    hab_coords = []
    for feature in hab_plots:
        try:
            coords = feature['geometry']['coordinates'][0]
            if isinstance(coords, list) and len(coords) >= 3 and all(isinstance(coord, list) and len(coord) == 2 for coord in coords):
                centroid_coords = np.mean(coords, axis=0).tolist()
                hab_coords.append(centroid_coords)
                hab_plot_data.append({
                    'centroid': centroid_coords,
                    'area': calculate_polygon_area(coords),
                    'vertices': coords
                })
            else:
                print(f"Warning: Skipping habitat plot with invalid coordinates: {feature['properties']['id']}")
        except (KeyError, TypeError) as e:
            print(f"Warning: Skipping habitat plot due to error: {e}, Plot ID: {feature.get('properties', {}).get('id', 'Unknown')}")
            continue

    water_coords = []
    for feature in water_plots:
        try:
            coords = feature['geometry']['coordinates'][0]
            if isinstance(coords, list) and len(coords) >= 3 and all(isinstance(coord, list) and len(coord) == 2 for coord in coords):
                centroid_coords = np.mean(coords, axis=0).tolist()
                water_coords.append(centroid_coords)
            else:
                print(f"Warning: Skipping water plot with invalid coordinates: {feature['properties']['id']}")
        except (KeyError, TypeError) as e:
            print(f"Warning: Skipping plot due to error: {e}, Plot ID: {feature.get('properties', {}).get('id', 'Unknown')}")
        continue

    hab_coords = np.array(hab_coords)
    water_coords = np.array(water_coords)

    output_features = []

    # Socio-economic factors (example values - these should ideally come from the input data or an external source)
    market_access = 0.7  # 0 to 1, higher is better
    community_support = 0.6 # 0 to 1, higher is better

    # Climate impact prediction (example values - these should ideally come from a climate model)
    predicted_temp_increase = 1.5 # Degrees Celsius increase in the next 10 years
    predicted_rainfall_change = -0.1 # Percentage decrease in rainfall in the next 10 years

    intervention_cost_per_area = 0.5  # Cost to implement margin or habitat per unit area
    yield_increase_per_intervention = 0.2 # Expected yield increase per unit intervention
    discount_rate = 0.05  # Discount rate for future yield

    mean_ag_yield = np.mean(ag_yields) if ag_yields.size else 0

    # Calculate average yield and soil quality for adaptive thresholds
    avg_yield = np.mean(ag_yields) if ag_yields.size else 0
    avg_soil_quality = np.mean([f['properties'].get('soil_quality', 0.5) for f in ag_plots])

    # Landscape Context: Habitat Scarcity
    total_habitat_area = sum([hab['area'] for hab in hab_plot_data]) if hab_plot_data else 0
    habitat_scarcity_factor = 1.0 - min(1.0, total_habitat_area / total_ag_area) if total_ag_area > 0 else 1.0

    for i, feature in enumerate(ag_plot_features):
        try:
            props = feature['properties']
            plot_id = props['id']
            plot_yield = props['yield']
            soil_quality = props.get('soil_quality', 0.5)
            pollinator_dependence = props.get('pollinator_dependence', 0.5)
            pest_pressure = props.get('pest_pressure', 0.5)
            slope = props.get('slope', 0.1)
            coords = feature['geometry']['coordinates'][0]
            plot_area = calculate_polygon_area(coords)
            plot_yield_density = plot_yield / plot_area if plot_area > 0 else 0
            yield_density_ratio = plot_yield_density / mean_yield_density if mean_yield_density > 0 else 1e-6

            ag_plot_centroid = ag_coords[i]

            min_distance_to_hab = np.min(cdist(ag_coords[i:i+1], hab_coords)) if hab_coords.size else np.inf
            min_distance_to_water = np.min(cdist(ag_coords[i:i+1], water_coords)) if water_coords.size else np.inf
            min_distance_to_hab_water = min(min_distance_to_hab, min_distance_to_water)

            # Landscape Connectivity Index
            connectivity_index = 0.0
            if hab_plot_data:
                for hab_item in hab_plot_data:
                    hab_centroid = hab_item['centroid']
                    hab_area = hab_item['area']
                    distance = math.sqrt((ag_plot_centroid[0] - hab_centroid[0])**2 + (ag_plot_centroid[1] - hab_centroid[1])**2)
                    connectivity_index += (hab_area / (distance + 0.0001))

            # Climate Risk Factor
            climate_risk_factor = (predicted_temp_increase * 0.6) - (predicted_rainfall_change * 0.4)

            # Socio-Economic Modifier
            socioeconomic_modifier = (market_access + community_support) / 2

            #Dynamic Ecosystem Services
            ecosystem_service_potential = 0.0
            if hab_plot_data:
                for hab_item in hab_plot_data:
                    hab_centroid = hab_item['centroid']
                    hab_area = hab_item['area']
                    distance = math.sqrt((ag_plot_centroid[0] - hab_centroid[0])**2 + (ag_plot_centroid[1] - hab_centroid[1])**2)
                    connectivity = sum([1 / (1 + math.sqrt((hab_centroid[0] - other_hab['centroid'][0])**2 + (hab_centroid[1] - other_hab['centroid'][1])**2))
                                        for other_hab in hab_plot_data if other_hab != hab_item])
                    ecosystem_service_potential += hab_area * math.exp(-distance / decay_constant) * (1 + connectivity)

            # Habitat Influence (Inverse Square Law)
            total_weighted_decay_factor = 0.0
            if hab_plot_data:
                for hab_item in hab_plot_data:
                    hab_centroid = hab_item['centroid']
                    hab_area = hab_item['area']
                    distance = calculate_distance(ag_plot_centroid, hab_centroid)
                    decay_factor = 1 / (distance**2 + 0.0001)
                    total_weighted_decay_factor += decay_factor * (hab_area / (total_ag_area + 0.0001))

            # Resource Synergy Score
            resource_synergy_score = (
                (1 / (yield_density_ratio + 0.0001)) *
                (soil_quality + 0.0001) *
                (1 + ecosystem_service_potential) *
                (1 + total_weighted_decay_factor) *
                (1 - min(1, min_distance_to_water / 0.1)) *
                (1 + (pollinator_dependence - 0.5) * 0.4) *
                (1 + (pest_pressure - 0.5) * 0.3) *
                (1 - slope)
            )

            # Area Ratio Factor
            area_ratio_factor = (plot_area / average_plot_area) if average_plot_area > 0 else 1.0
            area_ratio_factor = min(3.0, max(0.33, area_ratio_factor))
            resource_synergy_score_adjusted = resource_synergy_score + (area_ratio_factor - 1) * 0.3

            # Habitat Adjacency Score
            adjacency_score = 0.0
            if hab_coords.size > 0:
                distances_to_hab = cdist(ag_coords[i:i+1], hab_coords)
                adjacent_hab_indices = np.where(distances_to_hab < math.sqrt(average_plot_area / math.pi))[1]
                total_adjacent_hab_area = sum([hab_plot_data[j]['area'] for j in adjacent_hab_indices])
                adjacency_score = min(1.0, total_adjacent_hab_area / (plot_area + 0.0001))

            margin_intervention = 0.0
            habitat_conversion = 0.0

            # Base intervention level on synergy
            if resource_synergy_score_adjusted > 5:
                intervention_level = min(1.0, 0.6 + (resource_synergy_score_adjusted - 5) * 0.01)
            elif resource_synergy_score_adjusted > 2:
                intervention_level = min(0.4, 0.1 + (resource_synergy_score_adjusted - 2) * 0.1)
            elif resource_synergy_score_adjusted > 0.7:
                intervention_level = min(0.7, 0.1 + (resource_synergy_score_adjusted - 0.7) * 0.2)
            else:
                intervention_level = min(0.3, resource_synergy_score_adjusted * 0.25)

            # Area adjustment for intervention level.
            intervention_level = intervention_level * (plot_area / 150)

            # NPV proxy calculation with Time-Varying Yield
            cost_margin = plot_area * intervention_level * intervention_cost_per_area * 0.3
            cost_habitat = plot_area * intervention_level * intervention_cost_per_area

            npv_margin = 0
            npv_habitat = 0

            for year in range(1, 11):
                yield_increase_factor = max(0.05, 1 - (year - 1) * 0.03)
                yield_increase_margin = plot_yield * intervention_level * yield_increase_per_intervention * 0.6 * yield_increase_factor
                water_proximity_factor = 1 + (1 - min(1, min_distance_to_water / 0.2)) * 0.2
                yield_increase_margin *= water_proximity_factor
                yield_increase_habitat = plot_yield * intervention_level * yield_increase_per_intervention * yield_increase_factor

                npv_margin += yield_increase_margin / (1 + discount_rate)**year
                npv_habitat += yield_increase_habitat / (1 + discount_rate)**year

            npv_margin -= cost_margin
            npv_habitat -= cost_habitat

            # Adaptive Intervention Decision
            if climate_risk_factor > 0.5:  # Higher climate risk
                if habitat_scarcity_factor > 0.5:  #Scarcity, prioritize habitat
                    if npv_habitat * (1 + adjacency_score * 0.3) > npv_margin:
                        habitat_conversion = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        margin_intervention = 0.0
                    else:
                        margin_intervention = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        habitat_conversion = 0.0
                else: #Habitat not scarce, focus on margin improvements
                    if npv_margin > npv_habitat * (1 + adjacency_score * 0.3):
                        margin_intervention = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        habitat_conversion = 0.0
                    else:
                        habitat_conversion = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        margin_intervention = 0.0
            else:  # Lower climate risk
                if habitat_scarcity_factor > 0.5:  #Scarcity, prioritize habitat
                    if npv_habitat * (1 + adjacency_score * 0.3) > npv_margin:
                        habitat_conversion = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        margin_intervention = 0.0
                    else:
                        margin_intervention = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        habitat_conversion = 0.0
                else: #Habitat not scarce, focus on margin improvements
                    if npv_margin > npv_habitat * (1 + adjacency_score * 0.3):
                        margin_intervention = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        habitat_conversion = 0.0
                    else:
                        habitat_conversion = intervention_level * (1 + socioeconomic_modifier * 0.2)
                        margin_intervention = 0.0

            #Differential treatment of marginal land
            if min_distance_to_hab_water > 0.2:
                margin_intervention = min(1.0, margin_intervention * 1.5)

            #Connectivity Factor
            connectivity_factor = 0.0
            if habitat_conversion > 0 and hab_plot_data:
                ag_plot_vertices = feature['geometry']['coordinates'][0]
                for hab_item in hab_plot_data:
                    hab_vertices = hab_item['vertices']
                    for vertex in ag_plot_vertices:
                        if is_within_bounds(vertex, hab_vertices):
                            connectivity_factor = 0.5
                            break
                    if connectivity_factor > 0:
                        break

            habitat_conversion = min(1.0, habitat_conversion + connectivity_factor)


            if margin_intervention > 0 and habitat_conversion > 0:
                if margin_intervention * 1.15 >= habitat_conversion:
                    habitat_conversion = 0
                else:
                    margin_intervention = 0

            # Yield Modulation
            yield_factor = plot_yield / mean_ag_yield if mean_ag_yield > 0 else 1
            margin_intervention = margin_intervention * (1 + 0.08 * yield_factor)
            habitat_conversion = habitat_conversion * (1 - 0.04 * yield_factor)


            # Intervention Balancing
            total_intervention = margin_intervention + habitat_conversion
            if total_intervention > 1:
                margin_intervention = margin_intervention / total_intervention
                habitat_conversion = habitat_conversion / total_intervention


            output_features.append({
                "type": "Feature",
                "properties": {
                    "id": plot_id,
                    "margin_intervention": min(1.0, max(0.0, margin_intervention)),
                    "habitat_conversion": min(1.0, max(0.0, habitat_conversion))
                },
                "geometry": feature['geometry']
            })
        except (KeyError, IndexError, TypeError, RuntimeWarning, ZeroDivisionError) as e:
            print(f"Warning: Skipping plot due to error during processing: {e}, Plot ID: {feature.get('properties', {}).get('id', 'Unknown')}")
            continue

    for feature in other_plots:
        plot_id = feature['properties']['id']
        output_features.append({
            "type": "Feature",
            "properties": {
                "id": plot_id,
                "margin_intervention": 0,
                "habitat_conversion": 0
            },
            "geometry": feature['geometry']
        })

    output_geojson = {
        "type": "FeatureCollection",
        "features": output_features
    }

    with open('output.geojson', 'w') as f:
        json.dump(output_geojson, f, indent=2)

    return "Enhanced integrated heuristic interventions saved to output.geojson"


if __name__ == '__main__':
    result = enhanced_integrated_heuristic()
    print(result)