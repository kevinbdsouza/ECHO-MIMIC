import os
import json
import geopandas as gpd
from echo_mimic.config import Config
import numpy as np
import random


def find_touching_neighbors(target_farm_id, farm_geojson_path, column_id):
    # Load the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(farm_geojson_path)

    # Ensure the 'farm_id' exists in the GeoJSON properties
    if column_id not in gdf.columns:
        raise ValueError(f"The GeoJSON file does not have a {column_id} property.")

    # Get the geometry of the target farm_id
    target_farm = gdf[gdf[column_id] == target_farm_id]
    if target_farm.empty:
        raise ValueError(f"Farm ID {target_farm_id} not found in the GeoJSON file.")

    # Find the neighbors that touch the target farm
    neighbors = gdf[gdf.geometry.touches(target_farm.iloc[0].geometry)]

    # Extract the farm IDs of the neighbors
    neighbor_farm_ids = neighbors[column_id].tolist()
    print(f"The immediate touching neighbors of farm_id {target_farm_id} are: {neighbor_farm_ids}")
    return neighbor_farm_ids


def create_graph_prompt_file(current_farm_id, all_farms_geojson_path, farms_dir, use_hint=False):
    neighbor_farm_ids = find_touching_neighbors(current_farm_id, all_farms_geojson_path, column_id="id")

    prompt_template = (
        "Instructions: Your task is to decide which interventions need to be done at which agricultural plots "
        "(crops, type='ag_plot') and how to place them geometrically based on how the interventions increase ecological connectivity, "
        "while not decreasing NPV from a baseline value. The choice is between margin "
        "(convert only the margins) and habitat (convert a contiguous region) interventions. "
        "The margin interventions are chosen among the following directions on the boundary: "
        "north-east, north-west, south-east, south-west. The habitat conversions "
        "are chosen among the same directions in the internal area of polygons. "
        "The directions are computed by running a horizontal and a vertical line through the centre of each plot, and "
        "choosing them if they have interventions (as computed by IPOPT optimization) greater than a threshold. "
        "Existing habitat plots (type='hab_plots') remain unaffected. "
        "Integral index of connectivity (IIC) is used as the metric for ecological connectivity, which tries to increase the "
        "size of the connected components in the neighbourhood. It promotes fractions touching each other and extending the "
        "connectivity between existing habitats in the landscape, which includes the farm and its neighbours. "
        "There is a tradeoff between maximizing connectivity and maintaining NPV. "
        "The NPV is calculated based on how the interventions affect pollination and pest control "
        "services over distance and time, and how these affect yield. There is also the tradeoff between the cost of implementation and "
        "maintenance vs the benefit of increased yield. I will show you examples of initial input geojson and the output "
        "interventions and geometries suggested by IPOPT optimization for your neighbouring farms. In the input geojson, the id for "
        "each plot is in 'id', land use class in 'label', whether ag_plot ot hab_plot in 'type', yield in 'yield, the "
        "neighboring plot ids in 'nbs', and plot polygon in 'geometry'. "
        "In the output, 'plot_id' refers to the plot id, 'plot_type' refers to whether ag_plot ot hab_plot, "
        "'label' refers to land use class, 'margin_directions' and 'habitat_directions' show the margin intervention"
        " and habitat conversion directions on the boundary and internally respectively.\n\n"

        "Given these examples, provided input geojson of your farm, "
        "you need to predict the final intervention directions using heuristics created in python. "
        "Use names of margin_directions and habitat_directions for the predicted intervention geometries in the output, "
        "and don't alter these variable names. "
        "Look at properties that you think have a pattern (like yield, label, type, nbs, geometry), "
        "and the relative positioning of both the farm neighbours with respect to your farm and "
        "the plots with respect to each other within the farm in the context of ecological connectivity. "
        "Use all the given geometry information to infer these geographical relationships. "
        "Do not use plot_ids in the if statements as rules to assign directions. "
        "You can incorporate the parameters like crop prices and implementation and maintenance costs "
        "provided at the end in your heuristics. ")

    if use_hint:
        prompt_template = prompt_template + "You can also use the hint provided at the end. \n\n Data:\n\n"
    else:
        prompt_template = prompt_template + "\n\n Data:\n\n"

    neighbor_data = ""
    for idx, nb_farm_id in enumerate(neighbor_farm_ids, start=1):
        input_geojson_path = os.path.join(farms_dir, f"farm_{nb_farm_id}", "connectivity",
                                          "input.geojson")
        output_geojson_path = os.path.join(farms_dir, f"farm_{nb_farm_id}", "connectivity",
                                           "output_gt_directions.json")

        with open(input_geojson_path, "r") as input_file:
            input_geojson_content = input_file.read()

        with open(output_geojson_path, "r") as output_file:
            output_geojson_content = output_file.read()

        neighbor_data += (
            f"Neighbour {idx}: input: {input_geojson_content}\n\n"
            f"Output: {output_geojson_content}\n\n"
        )

    current_farm_input_path = os.path.join(farms_dir, f"farm_{current_farm_id}", "connectivity",
                                           "input.geojson")
    output_prompt_file = os.path.join(farms_dir, f"farm_{current_farm_id}", "connectivity",
                                      "prompt_input.txt")

    with open(current_farm_input_path, "r") as current_farm_file:
        current_farm_input_content = current_farm_file.read()

    current_farm_data = (
        f"Your farm: input: {current_farm_input_content}\n\n"
    )

    final_instructions = (
        "Final Instructions: I want you to infer the logic from the examples and work through the inferred logic to "
        "predict intervention directions. "
        "Give me your best estimate as to where the interventions should be placed at each agricultural plot. "
        "Proceed based on the conceptual framework inferred from provided examples. Do not hallucinate. Come up with heuristics. "
        "Don't provide the full json but rather provide the python code to produce the json using the decided heuristics. "
        "Explain your reasoning and think step by step before providing the code. "
        "Handle all the features, i.e., plot ids. "
        "Don't create new variable names. Use margin_directions and habitat_directions for the predicted intervention geometries in the output. "
        "In the code no need to define the input json again, just load it from the file input.geojson."
        "Save outputs to output.json. \n\n"
    )

    # optim_python_instructions = (
    #    f"This is the original CMA-ES python implementation that arrived at the interventions: {optim_python}\n\n"
    # )

    # params_instructions = (
    #    f"These are the optimization parameters used in the original optimization, including prices and costs: {params}\n\n"
    # )

    params_instructions = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}.\n\n"
    )

    current_farm_gt_path = os.path.join(farms_dir, f"farm_{current_farm_id}", "connectivity",
                                        "output_gt_directions.json")
    with open(current_farm_gt_path, "r") as output_file:
        current_farm_gt = output_file.read()
    hint = (
        f"Your hint is that the ground truth output.json for your farm is this: {current_farm_gt}. "
        "Don't copy these directly, and don't assign them by plot id, they are just to help you "
        "compare and come up with the right heuristics. "
    )

    # Combine all parts into the full prompt
    if use_hint:
        full_prompt = prompt_template + neighbor_data + current_farm_data + final_instructions + params_instructions + hint
    else:
        full_prompt = prompt_template + neighbor_data + current_farm_data + final_instructions + params_instructions

    # Write the prompt to the output file
    with open(output_prompt_file, "w") as file:
        file.write(full_prompt)


def create_farm_prompt_file_2(current_farm_id, all_farms_geojson_path, farms_dir):
    neighbor_farm_ids = find_touching_neighbors(current_farm_id, all_farms_geojson_path, column_id="id")

    prompt_template = (
        "Instructions: Your task is to decide which interventions need to be done at which agricultural plots "
        "(crops, type='ag_plot') based on how the interventions affect NPV. The choice is between margin "
        "(convert only the margins) and habitat (convert a contiguous region) interventions. "
        "The interventions can be fractional. Existing habitat plots (type='hab_plots') "
        "remain unaffected. The NPV is calculated based on how the interventions affect pollination and pest control "
        "services over distance and time, and how these affect yield. There is a tradeoff between the cost of implementation and "
        "maintenance vs the benefit of increased yield. I will show you examples of initial input geojson and the output "
        "interventions suggested by Pyomo optimization for your neighbouring farms. In the input geojson, the id for "
        "each plot is in 'id', land use class in 'label', whether ag_plot ot hab_plot in 'type', yield in 'yield, and "
        "and polygon in 'geometry'. In the output, only plots having non-zero interventions "
        "are shown, and the rest of the plots have zero interventions. Given these examples, provided input geojson of your farm, "
        "you need to predict the final interventions using heuristics created in python. "
        "Look at properties that you think have a pattern (like yield, label, type, geometry, do not use plot_ids to assign rules). "
        "You can compute metrics using these variables and others, and can even look at the graphical structure of the farms. "
        "You can incorporate the parameters like crop prices and implementation and maintenance costs "
        "provided at the end in your heuristics. \n\n"
        "Data:\n\n"
    )

    neighbor_data = ""
    for idx, nb_farm_id in enumerate(neighbor_farm_ids, start=1):
        input_geojson_path = os.path.join(farms_dir, f"farm_{nb_farm_id}",
                                          "input.geojson")
        output_geojson_path = os.path.join(farms_dir, f"farm_{nb_farm_id}",
                                           "output_gt.geojson")

        with open(input_geojson_path, "r") as input_file:
            input_geojson_content = input_file.read()

        with open(output_geojson_path, "r") as output_file:
            output_geojson_content = output_file.read()

        neighbor_data += (
            f"Neighbour {idx}: input: {input_geojson_content}\n\n"
            f"Output: {output_geojson_content}\n\n"
        )

    current_farm_input_path = os.path.join(farms_dir, f"farm_{current_farm_id}",
                                           "input.geojson")
    output_prompt_file = os.path.join(farms_dir, f"farm_{current_farm_id}",
                                      "prompt_input.txt")

    with open(current_farm_input_path, "r") as current_farm_file:
        current_farm_input_content = current_farm_file.read()

    current_farm_data = (
        f"Your farm: input: {current_farm_input_content}\n\n"
    )

    final_instructions = (
        "Final Instructions: I want you to infer the logic from the examples and work through the inferred logic to predict interventions. "
        "Give me your best estimate as to what fraction of which intervention should be done at each agricultural plot. "
        "Proceed based on the conceptual framework inferred from provided examples. Do not hallucinate. Come up with heuristics. "
        "Don't provide the full json but rather provide the python code to produce the json using the decided heuristics. "
        "Explain your reasoning and think step by step before providing the code. "
        "Handle all the features, i.e., plot ids. "
        "Don't create new variable names. Use margin_intervention and habitat_conversion for predicted values in the output. "
        "In the code no need to define the input json again, just load it from the file input.geojson."
        "Save outputs to output.geojson. \n\n"
    )

    # optim_python_instructions = (
    #    f"This is the original CMA-ES python implementation that arrived at the interventions: {optim_python}\n\n"
    # )

    params_instructions = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}"
    )

    # Combine all parts into the full prompt
    full_prompt = prompt_template + neighbor_data + current_farm_data + final_instructions + params_instructions

    # Write the prompt to the output file
    with open(output_prompt_file, "w") as file:
        file.write(full_prompt)


def create_farm_prompt_file(current_farm_id, all_farms_geojson_path, farms_dir):
    neighbor_farm_ids = find_touching_neighbors(current_farm_id, all_farms_geojson_path, column_id="farm_id")
    empty_outs = 0
    if len(neighbor_farm_ids) > 0:
        for n_id in neighbor_farm_ids:
            farm_path = os.path.join(farms_dir, f"farm_{n_id}")
            if not os.path.exists(farm_path):
                empty_outs += 1
                continue
            output_gt_path = os.path.join(farm_path, "output_gt.geojson")
            with open(output_gt_path, "r") as f:
                output_gt = json.load(f)
            if len(output_gt["features"]) == 0:
                empty_outs += 1

    print("Sampling random farms")
    while len(neighbor_farm_ids) < 3 or empty_outs > len(neighbor_farm_ids) - 2:
        r = random.randint(1, 5494)
        farm_path = os.path.join(farms_dir, f"farm_{r}")
        if r == current_farm_id or not os.path.exists(farm_path):
            continue
        output_gt_path = os.path.join(farm_path, "output_gt.geojson")
        with open(output_gt_path, "r") as f:
            output_gt = json.load(f)
        if len(output_gt["features"]) == 0:
            continue
        neighbor_farm_ids.append(r)

    prompt_template = (
        "Instructions: Your task is to decide which interventions need to be done at which agricultural plots "
        "(crops, type='ag_plot') based on how the interventions affect NPV. The choice is between margin "
        "(convert only the margins) and habitat (convert a contiguous region) interventions. The interventions can be fractional. "
        "Existing habitat plots (type='hab_plots;) remain unaffected. "
        "The NPV is calculated based on how the interventions affect pollination and pest control "
        "services over distance and time, and how these affect yield. There is a tradeoff between the cost of implementation and "
        "maintenance vs the benefit of increased yield. I will show you examples of initial input geojson and the output "
        "interventions suggested by Pyomo optimization for your immediate neighbouring farms or some other neighbouring farms "
        "in the landscape if you have no immediate neighbours. "
        "In the input json, the id for each plot is in 'id', land use class in 'label', whether ag_plot ot hab_plot in 'type', yield in 'yield, "
        "and the neighboring plot ids are given in 'nbs'. In the output, only plots having non-zero interventions "
        "are shown, and the rest of the plots have zero interventions. Given these examples, provided input geojson of your farm, "
        "you need to predict the final interventions using heuristics created in python. "
        "Look at properties that you think have a pattern (like yield, label, type, nbs, do not use plot_ids to assign rules). "
        "You can compute metrics using these variables and others, and can even look at the graphical structure of the farms. "
        "You can incorporate the parameters like crop prices and implementation and maintenance costs "
        "provided at the end in your heuristics. \n\n"
        "Data:\n\n"
    )

    neighbor_data = ""
    for idx, nb_farm_id in enumerate(neighbor_farm_ids, start=1):
        if not os.path.exists(os.path.join(farms_dir, f"farm_{nb_farm_id}")):
            continue
        input_geojson_path = os.path.join(farms_dir, f"farm_{nb_farm_id}",
                                          "input.geojson")
        output_geojson_path = os.path.join(farms_dir, f"farm_{nb_farm_id}",
                                           "output_gt.geojson")

        with open(input_geojson_path, "r") as input_file:
            input_geojson_content = input_file.read()

        with open(output_geojson_path, "r") as output_file:
            output_geojson_content = output_file.read()

        neighbor_data += (
            f"Neighbour {idx}: input: {input_geojson_content}\n\n"
            f"Output: {output_geojson_content}\n\n"
        )

    current_farm_input_path = os.path.join(farms_dir, f"farm_{current_farm_id}",
                                           "input.geojson")
    output_prompt_file = os.path.join(farms_dir, f"farm_{current_farm_id}",
                                      "prompt_input.txt")

    with open(current_farm_input_path, "r") as current_farm_file:
        current_farm_input_content = current_farm_file.read()

    current_farm_data = (
        f"Your farm: input: {current_farm_input_content}\n\n"
    )

    # geometry_info = ("if you want geometry information of your farm, you can load it from geometry.geojson, and it looks like this: "
    #                 "{'type': 'FeatureCollection', 'name': '', 'crs': {'type': 'name', 'properties': "
    #                 "{ 'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}, 'features': [{'type': 'Feature', 'properties': "
    #                 "{ 'habitat_id': null, 'label': 'Soybeans', 'type': 'ag_plot', 'yield': 0.0, 'id': 0 }, "
    #                 "'geometry': { 'type': 'Polygon', 'coordinates': [ [ [ -106935.0, 1116390.0 ], ... ] ] } }, .. ] ] } } ] } \n\n")

    final_instructions = (
        "Final Instructions: I want you to infer the logic from the examples and work through the inferred logic to predict interventions. "
        "Give me your best estimate as to what fraction of which intervention should be done at each agricultural plot. "
        "Proceed based on the conceptual framework inferred from provided examples. Do not hallucinate. Come up with heuristics. "
        "Don't provide the full json but rather provide the python code to produce the json using the decided heuristics. "
        "Explain your reasoning and think step by step before providing the code. "
        "Handle all the plot ids, i.e., plot ids. "
        "Don't create new variable names. Use margin_intervention and habitat_conversion for predicted values in the output. "
        "In the code no need to define the input json again, just load it from the file input.geojson."
        "Save outputs to output.geojson.\n\n"
    )

    params_instructions = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}."
    )

    # Combine all parts into the full prompt
    full_prompt = prompt_template + neighbor_data + current_farm_data + final_instructions + params_instructions

    # Write the prompt to the output file
    with open(output_prompt_file, "w") as file:
        file.write(full_prompt)


def create_plot_prompt_file(cfg, current_farm_id, current_plot_id):
    """
    Create a prompt_input.txt file with the specified template and provided farm and neighbor details.

    Args:
        current_farm_id (int): The farm ID of the current farm.
        current_plot_id (int): The plot ID.
    """

    prompt_template = (
        "Instructions: You will be given a geojson containing a crop plot (class=”central”) and its neighbours "
        "(class=\"nbs\"). Each plot contains its ids, some properties, and its geometry given by a multipolygon. "
        "You will also be given the fraction of interventions (margin intervention and habitat conversion) done in the "
        "central plot. Your task is to come up with heuristics using the inputs provided "
        "(the properties and shapes of the plots given) to compute the intervention "
        "fractions done in the central plot as the output. You can't use the given values of interventions "
        "(margin_intervention and habitat_conversion) to produce the output, they are only for reference to produce the heuristics. "
        "Provide the python code containing heuristics to generate the output dictionary. "
        "Don't create new variable names. Use margin_intervention and habitat_conversion for predicted values in the output. "
        "You can load the input geojson from input.geojson. You can only use this input data and no other data. "
        "Save outputs to output.geojson. \n\n"
    )

    input_data = ""
    plot_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{current_farm_id}",
                                     "plots_geojsons",
                                     f"plot_{current_plot_id}", "input.geojson")
    intervention_path = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{current_farm_id}",
                                     "plots_geojsons",
                                     f"plot_{current_plot_id}", "interventions.json")

    with open(plot_geojson_path, "r") as input_file:
        input_geojson_content = input_file.read()

    with open(intervention_path, "r") as input_file:
        interventions = input_file.read()

    input_data += "Data:\n" + f"Input: {input_geojson_content}\n Interventions: {interventions}\n"

    final_instructions = (
        "Final Instructions: I want you to infer the logic from the input given and work through the inferred logic to predict interventions. "
        "Give me your best estimate as to what fraction of each intervention should be done at the central plot. "
        "Proceed based on the conceptual framework inferred from provided examples. Do not hallucinate. Come up with heuristics. "
        "Make sure the heuristics you come up with try to reproduce the values of interventions you see in the input. "
        "Look at properties that you think have a pattern (the properties and shapes of the plots given). "
        "Explain your reasoning and think step by step."
    )

    # Combine all parts into the full prompt
    full_prompt = prompt_template + input_data + final_instructions

    output_prompt_file = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{current_farm_id}",
                                      "plots_geojsons",
                                      f"plot_{current_plot_id}", "prompt_input.txt")

    # Write the prompt to the output file
    with open(output_prompt_file, "w") as file:
        file.write(full_prompt)


def create_landscape_prompt_file(cfg, farm_id):
    input_path = os.path.join(landscape_dir, str(pos_id), "input.json")
    with open(input_path, "r") as input_file:
        input = json.load(input_file)

    prompt_template = (
        "Instructions: Your task is to make a land use decision at the 'central' position. "
        "Your options are between 'wheat', 'oat', 'habitat', 'corn', and 'soy'. "
        "In the data provided, you are given the position of your polygon as 'central' along with other properties like "
        "'wheat_yield', 'oat_yield', 'ecological_connectivity', 'corn_yield', and 'soy_yield'. The same properties are "
        "given for your immediate neighbours (position: north, south, east, west, north-east, north-west, south-east, south-west), and "
        "for neighbours of neighbours, if and when they exist. The positions of second-level neighbours are specified as d1-d2, "
        "where d1 is the direction from the central polygon to the immediate neighbour whose centroid is closest to the "
        "second-level neighbour, and d2 is the direction from that immediate neighbour to the second-level neighbour. "
        "This can also be thought of as a way to get to the second-level neighbour from the central polygon by "
        "moving in prescribed directions. crop yields are given in Tonnes/Hectare and 'ecological_connectivity' denotes how good the "
        "polygon is for animal movement and connectivity. When you make the land use decision of 'habitat', you increase the "
        "global landscape and neighbourhood connectivity, but loose the crop production in your polygon. "
        "Make the land use decision smartly in order to maximize ecological connectivity in the broader global landscape "
        "(this is not given to you) and not just your neighbourhood, given that you need to maintain a baseline level of "
        "total crop production across the broader landscape for every crop. "
        "Look at the given properties (crop yield, connectivity) for yourself and for your neighbours. "
        "One other caveat is that neighbours with zero yield for all crops are considered as already existing habitats. "
        "This can influence your decision, and can sway you either way depending on your context. "
        "You can compute metrics using given variables and others, and can even look at the graphical structure of the "
        "given neighbourhood. \n\n"
        f"Input Data: {input}\n\n"
    )

    output_prompt_file = os.path.join(landscape_dir, str(pos_id), "prompt_input.txt")
    with open(output_prompt_file, "w") as file:
        file.write(prompt_template)


def create_nudge_prompt_file(nudge_dir):
    heur_eco_intens_path = os.path.join(nudge_dir, "heuristics_gem_eco_intens.py")
    with open(heur_eco_intens_path, "r") as input_file:
        heur_eco_intens = input_file.read()

    heur_eco_conn_path = os.path.join(nudge_dir, "heuristics_gem_eco_conn.py")
    with open(heur_eco_conn_path, "r") as input_file:
        heur_eco_conn = input_file.read()

    params_instructions = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}."
    )

    prompt_template = (
        "Instructions: Your task is to come up with a message to the farmers in order to change their behaviour from "
        "following the ecological intensification heuristics that benefits solely their farm to following the "
        "ecological connectivity heuristics that increase landscape connectivity. "
        "Your communication to the farmer can be persuasive. It can provide incentives such as reducing the "
        "implementation or maintenance cost of an intervention by providing a one-time subsidy or yearly subsidies. "
        "It can compensate the farmers for yield that is lost to habitat conversion. It can communicate the benefits of "
        "landscape connectivity, and so on. \n\n"
        "The ecological intensification heuristics the farmer is currently following are:\n "
        f"```python\n{heur_eco_intens}\n``` \nThe ecological connectivity heuristics that you should nudge them towards are:\n "
        f"```python\n{heur_eco_conn}\n``` \n\n"
        f"The current parameters like prices and costs are: {params_instructions}\n\n"
        "One caveat is that the ecological connectivity heuristics are given in directions (margin_directions and "
        "habitat_directions), where there are 4 possible directions north-west, north-east, south-west, south-east. "
        "That means the resulting margin_intervention and habitat_conversion values can only be in multiples of 0.25 - "
        "0.25, 0.5, 0.75, 1. You need to convert the directions to these values. You can ignore the directions after that, "
        "and assume that the farmer will use whichever direction you want them to. "
        "Your goal should be to communicate a message that gets the farmer to alter the margin_intervention and "
        "habitat_conversion values for each of the plots, from the former to the latter.  "
        "Your final message to the farmer should be in this format \communication{message}. \n\n"
    )

    output_prompt_file = os.path.join(nudge_dir, "prompt_input.txt")
    with open(output_prompt_file, "w") as file:
        file.write(prompt_template)


# Example usage
if __name__ == "__main__":
    cfg = Config()
    # current_farm_id = 5
    # create_farm_prompt_file(cfg, current_farm_id)

    # current_plot_id = 0
    # create_plot_prompt_file(cfg, current_farm_id, current_plot_id)

    """
    farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "farms_s")
    all_farms_geojson_path = os.path.join(cfg.data_dir, "crop_inventory", "farms_m_s.geojson")
    farm_ids = np.arange(1, 5495)
    for farm_id in farm_ids:
        print(f"Running farm_id:{farm_id}")
        farm_path = os.path.join(farm_dir, f"farm_{farm_id}")
        if not os.path.exists(farm_path):
            continue

        output_gt_path = os.path.join(farm_path, "output_gt.geojson")
        with open(output_gt_path, "r") as f:
            output_gt = json.load(f)
        output_gt["name"] = "output"
        with open(output_gt_path, "w") as out_f:
            json.dump(output_gt, out_f)

        create_farm_prompt_file(farm_id, all_farms_geojson_path, farm_dir)
    """

    """
    landscape_dir = os.path.join(cfg.data_dir, "biomass", "landscape")
    pos_ids = np.arange(0, 8724)
    for pos_id in pos_ids:
        create_landscape_prompt_file(pos_id, landscape_dir)
    """

    farm_ids = np.arange(6, 7)
    for farm_id in farm_ids:
        nudge_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id}", "nudge")
        create_nudge_prompt_file(nudge_dir)