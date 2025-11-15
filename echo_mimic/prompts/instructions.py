"""Shared prompt instruction strings used across evolutionary strategies."""

FARM_COMMON_TASK_INSTRUCTIONS = (
    "The python programs are trying to solve the task of deciding which interventions need to be done at which agricultural plots "
    "(crops, type='ag_plot') based on how the interventions affect NPV. The choice is between margin "
    "(convert only the margins) and habitat (convert a contiguous region) interventions. "
    "The interventions can be fractional. Existing habitat plots (type='hab_plots') "
    "remain unaffected. The NPV is calculated based on how the interventions affect pollination and pest control "
    "services over distance and time, and how these affect yield. There is a tradeoff between the cost of implementation and "
    "maintenance vs the benefit of increased yield."
)

FARM_PARAMS_INSTRUCTIONS = (
    "You can incorporate parameters like crop prices and implementation and maintenance costs "
    "provided here in your heuristics. These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
    "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
    "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
    "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}. \n\n"
)

FARM_HALSTEAD_INSTRUCTIONS = (
    "Generate python code with high halstead h1 metric, lower maintainability index, and "
    "high halstead difficulty. \n\n"
)

GRAPH_COMMON_TASK_INSTRUCTIONS = (
    "The python programs are trying to solve the task of deciding which interventions need to be done at which agricultural plots "
    "(crops, type='ag_plot') and how to place them geometrically based on how the interventions increase ecological connectivity, "
    "while not decreasing NPV from a baseline value. "
    "The choice is between margin (convert only the margins) and habitat (convert a contiguous region) interventions. "
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
    "NPV is calculated based on how the interventions affect pollination and pest control "
    "services over distance and time, and how these affect yield. There is also the tradeoff between the cost of implementation and "
    "maintenance vs the benefit of increased yield. Look at properties that you think have a pattern "
    "(like yield, label, type, nbs, geometry, do not use plot_ids to assign rules), "
    "and the relative positioning of both the farm neighbours with respect to your farm and "
    "the plots with respect to each other within the farm in the context of ecological connectivity. "
    "Use all the given geometry information to infer these geographical relationships. \n\n"
)

GRAPH_PARAMS_INSTRUCTIONS = (
    "You can incorporate parameters like crop prices and implementation and maintenance costs "
    "provided here in your heuristics. These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
    "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
    "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
    "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}. \n\n"
)

GRAPH_HALSTEAD_INSTRUCTIONS = (
    "Generate python code with high halstead metrics like h1, h2, N1, N2, volume, "
    "difficulty, length, effort, and vocabulary.\n\n"
)


__all__ = [
    "FARM_COMMON_TASK_INSTRUCTIONS",
    "FARM_PARAMS_INSTRUCTIONS",
    "FARM_HALSTEAD_INSTRUCTIONS",
    "GRAPH_COMMON_TASK_INSTRUCTIONS",
    "GRAPH_PARAMS_INSTRUCTIONS",
    "GRAPH_HALSTEAD_INSTRUCTIONS",
]
