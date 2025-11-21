import json
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import cascaded_union

# --- Heuristic Parameters derived from Context Analysis (Plan Steps 1, 3) ---
# Yields in context: [2.45, 2.68, 0.5, 2.59, 2.49, 1.69] (ag plots)
# Low yield threshold: Plots around 0.5 or 1.7 (Spring wheat/Oats) are potential candidates for habitat conversion if adjacency is right.
YIELD_THRESHOLD_LOW = 1.7  # Crops below this yield are considered low productivity.

# Cost proxies (No explicit costs given for margin vs habitat, assuming a relative balance)
# Margin intervention cost proxy is implicitly handled by targeting higher yield crops for improvement.
# Habitat conversion cost proxy is implicitly handled by favoring adjacency (Step 2).

# --- Contextual Adjacency/Spatial Check (Plan Step 2) ---
# Function to check spatial adjacency/proximity (simplified: using geometric intersection for boundary sharing)
def check_adjacency(ag_plot_geom, habitat_geoms):
    """Checks if the agricultural plot shares a boundary with any habitat plot."""
    ag_shape = shape(ag_plot_geom)
    for hab_geom in habitat_geoms:
        hab_shape = shape(hab_geom)
        # Check for intersection (including touching boundaries)
        if ag_shape.intersects(hab_shape):
            # To strictly check for boundary sharing (not just overlapping area, though intersection works for boundary check too)
            if ag_shape.touches(hab_shape) or ag_shape.intersection(hab_shape).area > 1e-9:
                return True
    return False

# --- Economic Parameters (Plan Step 4) ---
# Assume the following typical relative costs based on general knowledge, since explicit costs are missing:
# Margin improvement is generally cheaper/faster to implement than large-scale habitat conversion if land is productive.
# Habitat conversion is favored where land is unproductive AND biodiversity benefit is high (i.e., near existing habitat).
COST_MARGIN_RELATIVE = 1.0
COST_HABITAT_RELATIVE = 2.0 # Habitat conversion is assumed more costly/disruptive.

# --- Load Data ---
try:
    with open('input.geojson', 'r') as f:
        input_data = json.load(f)
    
    # Neighbor 1 data is provided in the prompt as context for deriving rules, 
    # but since we are executing for farm_3, we only need the context data 
    # to establish the initial rules derived in the planning phase.
    # We analyze the context data to set up the helper structures needed for the heuristic application.
    context_data = json.loads("""{"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"label": "Broadleaf", "type": "hab_plots", "id": 1, "yield": 0.5}, "geometry": {"type": "Polygon", "coordinates": [[[3.79092127444394, 47.507746709568664], [6.950503612786814, 45.411884318562954], [13.866297373640442, 35.0], [0.0, 35.0], [0.0, 49.04098380215751], [3.79092127444394, 47.507746709568664]]]}}, {"type": "Feature", "properties": {"label": "Corn", "type": "ag_plot", "id": 2, "yield": 2.4580683244799997}, "geometry": {"type": "Polygon", "coordinates": [[[14.828445951301166, 59.34576064956489], [14.253200310368763, 54.58772110806867], [12.46771157815166, 52.15464144729916], [3.79092127444394, 47.507746709568664], [0.0, 49.04098380215751], [0.0, 65.0], [8.147491200265964, 65.0], [14.828445951301166, 59.34576064956489]]]}}, {"type": "Feature", "properties": {"label": "Soybeans", "type": "ag_plot", "id": 3, "yield": 2.68434075533}, "geometry": {"type": "Polygon", "coordinates": [[[12.46771157815166, 52.15464144729916], [21.40741271330687, 52.72285339992867], [21.980226926930918, 48.203084832694], [6.950503612786814, 45.411884318562954], [3.79092127444394, 47.507746709568664], [12.46771157815166, 52.15464144729916]]]}}, {"type": "Feature", "properties": {"label": "Spring wheat", "type": "ag_plot", "id": 4, "yield": 0.5}, "geometry": {"type": "Polygon", "coordinates": [[[14.828445951301166, 59.34576064956489], [22.056147869568672, 56.06836203267646], [21.424734872401988, 52.907750524783054], [14.253200310368763, 54.58772110806867], [14.828445951301166, 59.34576064956489]]]}}, {"type": "Feature", "properties": {"label": "Barley", "type": "ag_plot", "id": 5, "yield": 2.59051019914}, "geometry": {"type": "Polygon", "coordinates": [[[14.253200310368763, 54.58772110806867], [21.424734872401988, 52.907750524783054], [21.40741271330687, 52.72285339992867], [12.46771157815166, 52.15464144729916], [14.253200310368763, 54.58772110806867]]]}}, {"type": "Feature", "properties": {"label": "Soybeans", "type": "ag_plot", "id": 6, "yield": 2.49122591337}, "geometry": {"type": "Polygon", "coordinates": [[[26.003191121761873, 43.50685071031314], [21.980226926930918, 48.203084832694], [21.40741271330687, 52.72285339992867], [21.424734872401988, 52.907750524783054], [22.056147869568672, 56.06836203267646], [31.097766960520058, 65.0], [40.0, 65.0], [30.0, 60.0], [30.0, 50.0], [34.71196632458596, 45.28803367541404], [26.003191121761873, 43.50685071031314]]]}}, {"type": "Feature", "properties": {"label": "Broadleaf", "type": "hab_plots", "id": 7, "yield": 1.6948931827}, "geometry": {"type": "Polygon", "coordinates": [[[22.056147869568672, 56.06836203267646], [14.828445951301166, 59.34576064956489], [8.147491200265964, 65.0], [31.097766960520058, 65.0], [22.056147869568672, 56.06836203267646]]]}}, {"type": "Feature", "properties": {"label": "Exposed land/barren", "type": "hab_plots", "id": 8, "yield": 2.2448700065200002}, "geometry": {"type": "Polygon", "coordinates": [[[26.003191121761873, 43.50685071031314], [34.71196632458596, 45.28803367541404], [40.0, 40.0], [45.0, 40.0], [45.0, 35.0], [21.587161043157852, 35.0], [26.003191121761873, 43.50685071031314]]]}}, {"type": "Feature", "properties": {"label": "Oats", "type": "ag_plot", "id": 9, "yield": 1.69809901566}, "geometry": {"type": "Polygon", "coordinates": [[[21.980226926930918, 48.203084832694], [26.003191121761873, 43.50685071031314], [21.587161043157852, 35.0], [13.866297373640442, 35.0], [6.950503612786814, 45.411884318562954], [21.980226926930918, 48.203084832694]]]}}]}""")
except FileNotFoundError:
    # This should not happen based on execution environment setup, but included for robustness.
    exit(1)

# 1. Pre-process Context Data (Neighbor 1 Input & Context) to find habitat geometries and average yields
habitat_geoms = []
ag_plots_context = []

# Use context data to define habitat topology and typical yields for rule derivation
for feature in context_data['features']:
    if feature['properties']['type'] == 'hab_plots':
        habitat_geoms.append(feature['geometry'])
    elif feature['properties']['type'] == 'ag_plot':
        ag_plots_context.append(feature['properties'])

# Helper function to get average yield for a crop type in the context
def get_avg_yield(label):
    yields = [p['yield'] for p in ag_plots_context if p['label'] == label and p['yield'] > 0]
    return sum(yields) / len(yields) if yields else 0

# Context Analysis Summary:
# Corn (2.46), Soybeans (2.68, 2.49), Barley (2.59) -> High Yield > 2.4
# Spring Wheat (0.5), Oats (1.69) -> Low Yield < 1.7 (matching YIELD_THRESHOLD_LOW)

# 2. Prepare Neighbor 1 data structure (Neighbor 1 is needed for the final comparison in Step 3/4, 
# but for this isolated run, we primarily apply the derived heuristic to Farm 3 input.)
# We assume the input for Farm 3 is structured similarly to Neighbor 1 input provided in the prompt body.
# Since the prompt asks to generate output based on the *Farm 3 input* using heuristics derived from *Context + Neighbor 1*, 
# we only need the processed habitat geometry derived from the Context set.

# --- Heuristic Application to Farm 3 Input ---

output_features = []
farm3_ag_plots = [f for f in input_data['features'] if f['properties']['type'] == 'ag_plot']

# Calculate an overall reference yield for margin intervention potential
all_ag_yields = [p['yield'] for p in ag_plots_context if p['properties']['type'] == 'ag_plot']
MAX_YIELD_REF = max(all_ag_yields) if all_ag_yields else 1.0

for feature in farm3_ag_plots:
    props = feature['properties']
    geom = feature['geometry']
    
    plot_id = props['id']
    label = props['label']
    yield_val = props['yield']
    
    # Initialize outputs
    margin_intervention = 0.0
    habitat_conversion = 0.0
    
    # --- Step 1 & 3: Yield Evaluation ---
    
    is_low_yield = yield_val < YIELD_THRESHOLD_LOW
    
    # Calculate relative yield for margin intervention potential (Step 1)
    # Margin intervention priority is higher for plots that are currently productive (Yield close to max/average high yield)
    margin_potential_score = yield_val / MAX_YIELD_REF
    
    # --- Step 2: Adjacency Check ---
    is_adjacent_to_habitat = check_adjacency(geom, habitat_geoms)
    
    # --- Step 4: Combined Heuristic Assignment ---
    
    if is_low_yield:
        # Low productivity plot (e.g., Spring Wheat, Oats in context, or similar yield here)
        if is_adjacent_to_habitat:
            # Prioritize Habitat Conversion (Plan Step 2 & 3 logic: low value land next to high value natural area)
            # Assign high conversion fraction (e.g., 0.8)
            habitat_conversion = 0.8
            # Margin intervention is suppressed or set low (e.g., 0.1)
            margin_intervention = 0.1
        else:
            # Low productivity, no adjacency benefit -> Do nothing (0.0) or very low intervention
            margin_intervention = 0.0
            habitat_conversion = 0.0
            
    else:
        # High productivity plot (Yield >= YIELD_THRESHOLD_LOW, e.g., Corn, Soybeans, Barley)
        # Prioritize Margin Intervention (Plan Step 1 & 3 logic: maximize returns on good land)
        
        # Assign margin intervention proportional to how much yield improvement is possible (using margin_potential_score as a proxy for benefit/cost ratio)
        margin_intervention = min(1.0, 0.4 + margin_potential_score * 0.6) # Scale to ensure some baseline margin work unless yield is extremely low (already handled above)
        
        # Habitat conversion is suppressed unless the plot is huge and borders habitat (we skip complex area calculation and suppress conversion for high-yield plots)
        habitat_conversion = 0.0 
        
        # Refinement: If margin intervention is high, ensure habitat conversion is zero.
        if margin_intervention > 0.7:
             habitat_conversion = 0.0

    # Final output construction
    if margin_intervention > 1e-6 or habitat_conversion > 1e-6:
        # Ensure values are capped [0, 1]
        final_margin = max(0.0, min(1.0, margin_intervention))
        final_habitat = max(0.0, min(1.0, habitat_conversion))
        
        output_feature = {
            "type": "Feature",
            "properties": {
                "id": plot_id,
                "margin_intervention": final_margin,
                "habitat_conversion": final_habitat
            },
            "geometry": geom
        }
        output_features.append(output_feature)

# Final Output Structure
output_geojson = {
    "type": "FeatureCollection",
    "features": output_features
}

# Write output file
with open('autogen_output.geojson', 'w') as f:
    json.dump(output_geojson, f, indent=2)