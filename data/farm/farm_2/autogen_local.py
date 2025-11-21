import json
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import cascaded_union

# --- PARAMETERS (Inferred/Hypothetical based on prompt structure) ---
# These represent generalized cost/benefit trade-offs where:
# Margin Intervention Cost ($C_M$): Medium cost, good local yield stability.
# Habitat Conversion Cost ($C_H$): High initial cost, high long-term service gain.
# Service Benefit Multiplier (S): Higher service benefit leads to favoring conversion.

YIELD_THRESHOLD_HIGH = 2.55  # Yield above which margin is strongly considered (based on avg context yield ~2.6)
YIELD_THRESHOLD_LOW = 2.20   # Yield below which conversion is strongly considered (based on avg context yield ~2.6)

# Habitat Value Ranking (Higher is better):
# Broadleaf (2.89), Grassland (2.10, 2.62), Soybeans (2.61, high yield)
# Barren (2.49, 2.54) -> These are low value/baseline for comparison.

# Adjacency/Contiguity Boost Factors (Inferred necessity for conversion)
CONTIGUITY_SCORE_THRESHOLD = 1.5 # Threshold for significant neighborhood enhancement

# --- Helper Functions ---

def calculate_intersection_area(geom1, geom2):
    """Calculates the intersection area between two Shapely geometries."""
    try:
        poly1 = shape(geom1)
        poly2 = shape(geom2)
        if poly1.is_valid and poly2.is_valid:
            return poly1.intersection(poly2).area
        return 0.0
    except Exception:
        return 0.0

def calculate_adjacency_score(ag_plot_geom, habitat_features):
    """
    Calculates a score based on how much the ag_plot borders high-value habitat.
    Adjacency is approximated by calculating the area of intersection
    (since we don't have explicit edge distance calculation here,
    we treat overlapping or highly proximal areas as 'adjacent' candidates).
    This score simulates proximity/contact benefit.
    """
    ag_poly = shape(ag_plot_geom)
    adjacency_score = 0.0
    
    # Define high-value habitats based on context yields (Broadleaf, High Yield Grassland/Soy)
    HIGH_VALUE_LABELS = ["Broadleaf", "Grassland"]
    
    for hab in habitat_features:
        if hab['properties']['type'] == 'hab_plots':
            hab_value = hab['properties']['yield']
            
            # Simple check: if habitat yield is high, or it's a known good type
            is_high_value = (hab_value > YIELD_THRESHOLD_HIGH) or (hab['properties']['label'] in HIGH_VALUE_LABELS)
            
            if is_high_value:
                intersection_area = calculate_intersection_area(ag_plot_geom, hab['geometry'])
                # Score heavily based on overlap area (proxy for strong adjacency/border contact)
                adjacency_score += intersection_area * (hab_value / 3.0) # Normalize value impact
                
    return adjacency_score

def check_contiguity_potential(ag_plot_geom, all_habitat_features):
    """
    Checks if converting this ag_plot would significantly increase habitat contiguity
    by merging it with existing habitat patches.
    Returns a score based on how many distinct habitat neighbors it touches/merges.
    """
    ag_poly = shape(ag_plot_geom)
    
    # Create a union of all existing habitat polygons
    habitat_polygons = []
    for hab in all_habitat_features:
        if hab['properties']['type'] == 'hab_plots':
            try:
                habitat_polygons.append(shape(hab['geometry']))
            except Exception:
                pass
    
    if not habitat_polygons:
        return 0.0

    existing_habitat_union = cascaded_union(habitat_polygons)
    
    if not existing_habitat_union.is_valid:
        return 0.0

    # Calculate the area of the new combined geometry if conversion happens
    potential_new_habitat = ag_poly.union(existing_habitat_union)
    
    # Contiguity Boost = (Area of Union) - (Area of Original Habitat)
    # If this result is close to the area of the ag_plot, conversion creates a large contiguous block.
    contiguity_gain = potential_new_habitat.area - existing_habitat_union.area
    
    # Normalize by the size of the plot itself to get a ratio/score
    try:
        plot_area = ag_poly.area
        if plot_area > 0:
            return contiguity_gain / plot_area
        return 0.0
    except:
        return 0.0

# --- Main Logic Execution ---

INPUT_FILE = 'input.geojson'
OUTPUT_FILE = 'autogen_output.geojson'

try:
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    # If running in the final environment, ensure the data is loaded correctly
    data = {"type": "FeatureCollection", "features": []}

ag_plots = [f for f in data['features'] if f['properties']['type'] == 'ag_plot']
all_features = data['features']

results = []

# 1. Pre-calculate global metrics needed for heuristics (Adjacency and Contiguity)
habitat_features = [f for f in all_features if f['properties']['type'] == 'hab_plots']
plot_metrics = {}

for ag_plot in ag_plots:
    plot_id = ag_plot['properties']['id']
    geom = ag_plot['geometry']
    yield_val = ag_plot['properties']['yield']
    
    adj_score = calculate_adjacency_score(geom, habitat_features)
    contiguity_score = check_contiguity_potential(geom, all_features)
    
    plot_metrics[plot_id] = {
        'yield': yield_val,
        'adj_score': adj_score,
        'contiguity': contiguity_score
    }

# 2. Apply Heuristics (Steps 3, 4, 5 from Plan)
for ag_plot in ag_plots:
    plot_id = ag_plot['properties']['id']
    metrics = plot_metrics[plot_id]
    
    Y_plot = metrics['yield']
    A_adj = metrics['adj_score']
    C_score = metrics['contiguity']
    
    margin_intervention = 0.0
    habitat_conversion = 0.0
    
    # --- Heuristic 3: Margin Intervention ---
    # High yield AND adjacent to good habitat -> Prioritize margin maintenance (Pollination/Pest)
    if Y_plot > YIELD_THRESHOLD_HIGH and A_adj > 0.1: # Use a small baseline adjacency requirement
        # High margin priority: 0.7 to 1.0
        margin_intervention = min(1.0, 0.5 + (A_adj / 5.0)) 
    
    # --- Heuristic 4: Habitat Conversion ---
    # Low yield OR high contiguity potential -> Strong candidate for conversion
    
    conversion_candidate = False
    
    if Y_plot < YIELD_THRESHOLD_LOW:
        # Low yield plots are prime conversion targets
        conversion_candidate = True
        conversion_priority = 0.8
        
    elif C_score > CONTIGUITY_SCORE_THRESHOLD:
        # Plot forms a crucial link for creating a large habitat block
        conversion_candidate = True
        # Priority scales with contiguity gain (Max 1.0 if C_score is high enough)
        conversion_priority = min(1.0, C_score / CONTIGUITY_SCORE_THRESHOLD * 0.7)
        
    if conversion_candidate:
        habitat_conversion = max(habitat_conversion, conversion_priority)

    # --- Heuristic 5: Tradeoff Finalization ---
    # If both are high, we must choose or blend. 
    # General Rule: Conversion is favored if yield is very low, even if margin benefits exist.
    
    if habitat_conversion > 0.7 and margin_intervention > 0.5:
        # Conflict: Low yield plot next to good habitat.
        if Y_plot < YIELD_THRESHOLD_HIGH * 0.95: # Favor conversion if yield is definitively low
            margin_intervention = 0.0 # Cancel margin in favor of full conversion
        else:
            # Moderate yield plot: Blend the services (e.g., Margin 0.5, Conversion 0.5)
            margin_intervention = max(0.0, margin_intervention - habitat_conversion * 0.5)
            habitat_conversion = max(0.0, habitat_conversion * 0.5)

    # Ensure values are strictly within [0, 1]
    margin_intervention = max(0.0, min(1.0, margin_intervention))
    habitat_conversion = max(0.0, min(1.0, habitat_conversion))

    # Create output feature
    output_feature = {
        "type": "Feature",
        "properties": {
            "id": ag_plot['properties']['id'],
            "margin_intervention": margin_intervention,
            "habitat_conversion": habitat_conversion
        },
        "geometry": ag_plot['geometry']
    }
    
    if margin_intervention > 0 or habitat_conversion > 0:
        results.append(output_feature)

# --- Final Output ---
final_output = {
    "type": "FeatureCollection",
    "features": results
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_output, f, indent=4)