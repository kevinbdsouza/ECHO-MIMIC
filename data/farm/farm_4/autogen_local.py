import json
from collections import defaultdict

# --- Heuristic Parameters derived from Context Analysis ---

# 1. Margin Intervention Heuristic (Based on low yield relative to crop type)
# Assume baseline yields for context crops (based on neighbor examples):
# Wheat/Oats baseline: ~1.6 (Low yield in context)
# Canola: ~2.8
# Soybeans: ~2.6
# Threshold for triggering high margin intervention: If yield is significantly below neighbor's average for that crop, or below a general absolute low threshold (0.5 in context).

# Heuristic 1: Trigger Margin Intervention if yield is very low (<= 0.5) OR if yield is less than 50% of the highest observed yield for that crop type in the dataset.
YIELD_ABSOLUTE_LOW_THRESHOLD = 0.5
YIELD_RELATIVE_THRESHOLD_FACTOR = 0.5

# 2. Habitat Conversion Heuristic (Based on low yield AND adjacency to existing habitat)
# Proximity check will be based on simple bounding box overlap/close centroid distance, as full geometry intersection is complex without external libraries.
# Since full geometry processing is restricted, we will rely on the *pattern observed*: low yield plots adjacent to habitat plots are candidates for conversion.
# In the context, habitat plot 4 is located centrally/to the East. Plots 2, 7, 8, 9 are near it. Plots 2 and 9 (low yield wheat) are strong candidates.

# Heuristic 2: A plot is a candidate for habitat conversion if its yield is very low (<= 1.0) AND it shares a boundary or is very close (within a determined proximity factor) to an existing habitat plot.

# Since we cannot robustly calculate adjacency without Shapely/GeoPandas, we must rely on a proxy:
# Proxy Heuristic: If yield <= 0.5 AND the plot is near the *region* occupied by habitats in the context data (e.g., high X coordinates in the context example).
# Given the highly constrained environment, we use the observed pattern for the context data where low yield plots (2, 9) near the habitat (4) are converted.

# For farm 4, we will look for the lowest yields overall to prioritize habitat conversion there, as implementation cost for habitat is often higher than margin treatments.

# 3. Cost/Benefit Tradeoff & Baseline (Yield vs. Intervention Type)
# If yield is high (e.g., > 2.0), default intervention is 0, unless a specific high-value crop benefits from minor margin improvement (not clearly indicated here, so we stick to 0 baseline).
# If yield is extremely low (<= 0.5), prioritize Habitat Conversion if geographically plausible (which we proxy by prioritizing the lowest yield plots overall).

# Proxy parameters (based on general farm economics often implied in these problems):
COST_MARGIN = 1.0  # Normalized cost of margin intervention
COST_HABITAT = 5.0 # Normalized cost of habitat intervention (higher cost)

# --- Helper Functions ---

def calculate_centroid(coords):
    """Approximation of polygon centroid."""
    x_coords = [p[0] for poly in coords for p in poly]
    y_coords = [p[1] for poly in coords for p in poly]
    if not x_coords:
        return (0.0, 0.0)
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def get_bounding_box(geometry):
    """Get min/max coordinates for simple proximity check."""
    if geometry['type'] == 'Polygon':
        all_coords = [c for poly in geometry['coordinates'] for c in poly]
        if not all_coords:
            return None
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        return (min(xs), max(xs), min(ys), max(ys))
    return None

def is_close_bbox(bbox1, bbox2, proximity_threshold=5.0):
    """Check if two bounding boxes are geographically close."""
    if not bbox1 or not bbox2:
        return False
    
    min_x1, max_x1, min_y1, max_y1 = bbox1
    min_x2, max_x2, min_y2, max_y2 = bbox2
    
    # Simple distance between closest edges (using max coordinate separation)
    x_dist = max(0, min_x1 - max_x2, min_x2 - max_x1)
    y_dist = max(0, min_y1 - max_y2, min_y2 - max_y1)
    
    # If they overlap or are very close (within threshold distance)
    return (x_dist + y_dist) <= proximity_threshold

# --- Main Execution ---

def determine_interventions(input_data):
    ag_plots = [f for f in input_data['features'] if f['properties']['type'] == 'ag_plot']
    hab_plots = [f for f in input_data['features'] if f['properties']['type'] == 'hab_plots']
    
    # 1. Pre-analysis: Calculate baseline statistics and bounding boxes
    
    # Calculate overall max yield per crop type for relative comparison
    crop_max_yield = defaultdict(float)
    plot_data = {}
    
    for plot in ag_plots:
        p = plot['properties']
        plot_id = p['id']
        label = p['label']
        yield_val = p['yield']
        
        crop_max_yield[label] = max(crop_max_yield[label], yield_val)
        
        plot_data[plot_id] = {
            'label': label,
            'yield': yield_val,
            'geometry': plot['geometry'],
            'bbox': get_bounding_box(plot['geometry']),
            'intervention': {'margin_intervention': 0.0, 'habitat_conversion': 0.0}
        }

    # Calculate bounding boxes for habitat plots
    hab_bboxes = [get_bounding_box(h['geometry']) for h in hab_plots]

    # 2. Heuristic Application
    
    # Step 4 Baseline: Default is 0 intervention.
    
    # First Pass: Determine Margin Intervention candidates (Step 1)
    for plot_id, data in plot_data.items():
        label = data['label']
        yield_val = data['yield']
        max_y = crop_max_yield[label]
        
        margin_intervention = 0.0
        
        # Heuristic 1: Trigger high margin intervention if yield is extremely low, or relative performance is poor.
        if yield_val <= YIELD_ABSOLUTE_LOW_THRESHOLD:
            # Very poor performance suggests high immediate need for margin treatment IF it can respond.
            # We set a moderate default intervention if it's at the absolute bottom, assuming high potential for response.
            margin_intervention = 0.4 
            
        elif max_y > 0 and yield_val < YIELD_RELATIVE_THRESHOLD_FACTOR * max_y:
            # Yield is less than 50% of the best in class for this crop.
            margin_intervention = 0.2
            
        data['intervention']['margin_intervention'] = margin_intervention

    # Second Pass: Determine Habitat Conversion candidates (Step 2 & 3)
    
    # Identify plots that are geographically near habitats (Proxy for adjacency)
    habitat_proximity_candidates = []
    for plot_id, data in plot_data.items():
        is_proximal = False
        for hab_bbox in hab_bboxes:
            if data['bbox'] and is_close_bbox(data['bbox'], hab_bbox, proximity_threshold=10.0):
                is_proximal = True
                break
        
        if is_proximal:
            habitat_proximity_candidates.append(plot_id)

    # Refine Interventions based on proximity and yield (Step 3 Tradeoff)
    
    # Sort candidates by lowest yield first to prioritize high-impact conversions (Step 3)
    sorted_candidates = sorted(
        [plot_id for plot_id in habitat_proximity_candidates], 
        key=lambda pid: plot_data[pid]['yield']
    )
    
    # Based on context, low yield wheat plots (0.5) near habitat were converted.
    # If yield is <= 0.5 AND proximal, maximize habitat conversion.
    
    for plot_id in sorted_candidates:
        data = plot_data[plot_id]
        yield_val = data['yield']
        
        if yield_val <= YIELD_ABSOLUTE_LOW_THRESHOLD:
            # Prioritize Habitat Conversion over Margin Intervention for very poor, proximal plots (Step 3)
            data['intervention']['habitat_conversion'] = 1.0
            # Cancel margin intervention if habitat conversion is maximized
            data['intervention']['margin_intervention'] = 0.0 
        elif yield_val <= 1.0:
            # Moderate yield loss near habitat -> Moderate conversion
            data['intervention']['habitat_conversion'] = 0.5
            
    # Final check: Ensure no plot has both high habitat conversion and high margin intervention unless explicitly intended.
    # (Handled by explicitly setting margin to 0.0 when habitat is maximized above).
    
    # 3. Construct Output GeoJSON
    output_features = []
    for plot in ag_plots:
        plot_id = plot['properties']['id']
        data = plot_data[plot_id]
        
        output_feature = {
            "type": "Feature",
            "geometry": plot['geometry'],
            "properties": {
                "id": plot_id,
                "margin_intervention": data['intervention']['margin_intervention'],
                "habitat_conversion": data['intervention']['habitat_conversion']
            }
        }
        output_features.append(output_feature)
        
    return {"type": "FeatureCollection", "features": output_features}


# --- Execution Start ---
try:
    with open('input.geojson', 'r') as f:
        input_data = json.load(f)
    
    output_data = determine_interventions(input_data)
    
    with open('autogen_output.geojson', 'w') as f:
        json.dump(output_data, f, indent=2)

except FileNotFoundError:
    # If running outside the intended execution environment and file is missing, we cannot proceed.
    pass
except json.JSONDecodeError:
    # Handle case where input.geojson is invalid JSON.
    pass
except Exception as e:
    # Catch unexpected errors during processing.
    pass