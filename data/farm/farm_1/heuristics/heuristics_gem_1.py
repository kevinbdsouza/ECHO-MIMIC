# Heuristic-based intervention predictor for SCALAR crops
# Strategy (inferred from examples):
# - hab_plots (existing habitat) remain unchanged (no interventions).
# - ag_plot: if the plot's bounding box overlaps with any habitat plot's bounding box (adjacent in a loose sense),
#   assign full habitat conversion (1.0) and full margin intervention (1.0).
# - otherwise, assign partial habitat conversion based on distance to the nearest habitat bbox center, and
#   a base margin that scales with plot area (normalized by the largest ag_plot area).
# - All plots carry their original attributes; ag_plots get margin_intervention and habitat_conversion fields.
# - Input is read from input.geojson and output written to output.geojson.

import json
import math
from typing import List, Tuple, Dict, Any

INPUT_GEOJSON_PATH = "input.geojson"
OUTPUT_GEOJSON_PATH = "output.geojson"

def extract_polygon_points(geom: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Extract a list of (x, y) points from a Polygon or MultiPolygon geometry.
    We take the outer ring of the first polygon for robustness (enough for a bbox/center).
    """
    if geom is None or "type" not in geom or "coordinates" not in geom:
        return []
    gtype = geom["type"]
    coords = geom["coordinates"]
    pts = []
    if gtype == "Polygon":
        # coords: [ [ [x1,y1], [x2,y2], ... ], [hole1], ... ]
        if isinstance(coords, list) and len(coords) > 0:
            ring = coords[0]
            for p in ring:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
    elif gtype == "MultiPolygon":
        # coords: [ [ [ [x1,y1], ... ] ], [ [ [x1,y1], ... ] ], ... ]
        if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list) and len(coords[0]) > 0:
            first_poly = coords[0]
            if isinstance(first_poly, list) and len(first_poly) > 0 and isinstance(first_poly[0], list):
                ring = first_poly[0]
                for p in ring:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
    return pts

def polygon_area(pts: List[Tuple[float, float]]) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5

def bbox_of_points(pts: List[Tuple[float, float]]):
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))

def center_of_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, maxx, miny, maxy = bbox
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

def bbox_overlap(b1, b2) -> bool:
    minx1, maxx1, miny1, maxy1 = b1
    minx2, maxx2, miny2, maxy2 = b2
    return (max(minx1, minx2) <= min(maxx1, maxx2)) and (max(miny1, miny2) <= min(maxy1, maxy2))

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    # Load input
    with open(INPUT_GEOJSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    # Gather habitat plots bounding boxes and centers
    hab_plots = []
    ag_plots = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        gtype = props.get("type", "")
        pts = extract_polygon_points(geom)
        bbox = bbox_of_points(pts)
        center = center_of_bbox(bbox)
        area = polygon_area(pts)

        entry = {
            "feature": feat,
            "props": props,
            "geom": geom,
            "pts": pts,
            "bbox": bbox,
            "center": center,
            "area": area
        }

        if gtype == "hab_plots":
            hab_plots.append(entry)
        elif gtype == "ag_plot":
            ag_plots.append(entry)
        else:
            # Unknown type; treat as non-interventionable (no interventions)
            ag_plots.append(entry)

    # Precompute max area among ag_plots for normalization
    max_area = 0.0
    for a in ag_plots:
        if a["area"] > max_area:
            max_area = a["area"]
    if max_area <= 0:
        max_area = 1.0  # avoid division by zero

    # Precompute habitat bbox centers for distance metric
    hab_centers = []
    hab_bboxes = []
    for h in hab_plots:
        hab_centers.append(h["center"])
        hab_bboxes.append(h["bbox"])

    # If no habitat plots exist, we'll handle gracefully
    has_habitat = len(hab_centers) > 0
    max_dist = 100.0  # scale for distance-based habitat conversion

    # Assign interventions
    for a in ag_plots:
        feat = a["feature"]
        props = feat.get("properties", {})
        gtype = props.get("type", "")

        if gtype != "ag_plot":
            # Not an ag_plot: leave interventions as 0
            props["margin_intervention"] = 0.0
            props["habitat_conversion"] = 0.0
            continue

        # Compute adjacency to habitat plots (loose bbox overlap)
        adj_to_hab = False
        for hb in hab_bboxes:
            if bbox_overlap(a["bbox"], hb):
                adj_to_hab = True
                break

        if adj_to_hab and has_habitat:
            margin = 1.0
            habitat_conv = 1.0
        else:
            # Distance-based and area-based heuristics
            area_norm = a["area"] / max_area  # 0..1
            if has_habitat:
                # distance to nearest habitat center
                dists = [distance(a["center"], hc) for hc in hab_centers]
                dist_to_hab = min(dists) if dists else max_dist
                dist_norm = min(1.0, dist_to_hab / max_dist)
                habitat_conv = max(0.0, 1.0 - dist_norm)
            else:
                habitat_conv = 0.0

            # Margin scaling with area, with a baseline
            margin = min(1.0, 0.6 + 0.4 * area_norm)
            # If habitat conversion is nonzero but not adjacent, allow margin to stay reasonably high
            if habitat_conv > 0.0 and margin < 0.9:
                margin = min(1.0, margin + 0.1 * habitat_conv)

        # Clip to [0,1]
        margin = max(0.0, min(1.0, margin))
        habitat_conv = max(0.0, min(1.0, habitat_conv))

        props["margin_intervention"] = float(round(margin, 12))
        props["habitat_conversion"] = float(round(habitat_conv, 12))

    # Build output FeatureCollection
    output_features = []
    for f in features:
        # Find corresponding updated feature object
        # We'll mutate in place by updating properties; keep geometry as is
        updated_feat = {
            "type": "Feature",
            "properties": f.get("properties", {}),
            "geometry": f.get("geometry", {})
        }
        output_features.append(updated_feat)

    output = {
        "type": "FeatureCollection",
        "name": "output_gt",
        "crs": data.get("crs", {}),
        "features": output_features
    }

    # Write output
    with open(OUTPUT_GEOJSON_PATH, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2)

if __name__ == "__main__":
    main()
# Heuristic-based intervention predictor for SCALAR crops
# Strategy (inferred from examples):
# - hab_plots (existing habitat) remain unchanged (no interventions).
# - ag_plot: if the plot's bounding box overlaps with any habitat plot's bounding box (adjacent in a loose sense),
#   assign full habitat conversion (1.0) and full margin intervention (1.0).
# - otherwise, assign partial habitat conversion based on distance to the nearest habitat bbox center, and
#   a base margin that scales with plot area (normalized by the largest ag_plot area).
# - All plots carry their original attributes; ag_plots get margin_intervention and habitat_conversion fields.
# - Input is read from input.geojson and output written to output.geojson.

import json
import math
from typing import List, Tuple, Dict, Any

INPUT_GEOJSON_PATH = "input.geojson"
OUTPUT_GEOJSON_PATH = "output.geojson"

def extract_polygon_points(geom: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Extract a list of (x, y) points from a Polygon or MultiPolygon geometry.
    We take the outer ring of the first polygon for robustness (enough for a bbox/center).
    """
    if geom is None or "type" not in geom or "coordinates" not in geom:
        return []
    gtype = geom["type"]
    coords = geom["coordinates"]
    pts = []
    if gtype == "Polygon":
        # coords: [ [ [x1,y1], [x2,y2], ... ], [hole1], ... ]
        if isinstance(coords, list) and len(coords) > 0:
            ring = coords[0]
            for p in ring:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
    elif gtype == "MultiPolygon":
        # coords: [ [ [ [x1,y1], ... ] ], [ [ [x1,y1], ... ] ], ... ]
        if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list) and len(coords[0]) > 0:
            first_poly = coords[0]
            if isinstance(first_poly, list) and len(first_poly) > 0 and isinstance(first_poly[0], list):
                ring = first_poly[0]
                for p in ring:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
    return pts

def polygon_area(pts: List[Tuple[float, float]]) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5

def bbox_of_points(pts: List[Tuple[float, float]]):
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))

def center_of_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, maxx, miny, maxy = bbox
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

def bbox_overlap(b1, b2) -> bool:
    minx1, maxx1, miny1, maxy1 = b1
    minx2, maxx2, miny2, maxy2 = b2
    return (max(minx1, minx2) <= min(maxx1, maxx2)) and (max(miny1, miny2) <= min(maxy1, maxy2))

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    # Load input
    with open(INPUT_GEOJSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    # Gather habitat plots bounding boxes and centers
    hab_plots = []
    ag_plots = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        gtype = props.get("type", "")
        pts = extract_polygon_points(geom)
        bbox = bbox_of_points(pts)
        center = center_of_bbox(bbox)
        area = polygon_area(pts)

        entry = {
            "feature": feat,
            "props": props,
            "geom": geom,
            "pts": pts,
            "bbox": bbox,
            "center": center,
            "area": area
        }

        if gtype == "hab_plots":
            hab_plots.append(entry)
        elif gtype == "ag_plot":
            ag_plots.append(entry)
        else:
            # Unknown type; treat as non-interventionable (no interventions)
            ag_plots.append(entry)

    # Precompute max area among ag_plots for normalization
    max_area = 0.0
    for a in ag_plots:
        if a["area"] > max_area:
            max_area = a["area"]
    if max_area <= 0:
        max_area = 1.0  # avoid division by zero

    # Precompute habitat bbox centers for distance metric
    hab_centers = []
    hab_bboxes = []
    for h in hab_plots:
        hab_centers.append(h["center"])
        hab_bboxes.append(h["bbox"])

    # If no habitat plots exist, we'll handle gracefully
    has_habitat = len(hab_centers) > 0
    max_dist = 100.0  # scale for distance-based habitat conversion

    # Assign interventions
    for a in ag_plots:
        feat = a["feature"]
        props = feat.get("properties", {})
        gtype = props.get("type", "")

        if gtype != "ag_plot":
            # Not an ag_plot: leave interventions as 0
            props["margin_intervention"] = 0.0
            props["habitat_conversion"] = 0.0
            continue

        # Compute adjacency to habitat plots (loose bbox overlap)
        adj_to_hab = False
        for hb in hab_bboxes:
            if bbox_overlap(a["bbox"], hb):
                adj_to_hab = True
                break

        if adj_to_hab and has_habitat:
            margin = 1.0
            habitat_conv = 1.0
        else:
            # Distance-based and area-based heuristics
            area_norm = a["area"] / max_area  # 0..1
            if has_habitat:
                # distance to nearest habitat center
                dists = [distance(a["center"], hc) for hc in hab_centers]
                dist_to_hab = min(dists) if dists else max_dist
                dist_norm = min(1.0, dist_to_hab / max_dist)
                habitat_conv = max(0.0, 1.0 - dist_norm)
            else:
                habitat_conv = 0.0

            # Margin scaling with area, with a baseline
            margin = min(1.0, 0.6 + 0.4 * area_norm)
            # If habitat conversion is nonzero but not adjacent, allow margin to stay reasonably high
            if habitat_conv > 0.0 and margin < 0.9:
                margin = min(1.0, margin + 0.1 * habitat_conv)

        # Clip to [0,1]
        margin = max(0.0, min(1.0, margin))
        habitat_conv = max(0.0, min(1.0, habitat_conv))

        props["margin_intervention"] = float(round(margin, 12))
        props["habitat_conversion"] = float(round(habitat_conv, 12))

    # Build output FeatureCollection
    output_features = []
    for f in features:
        # Find corresponding updated feature object
        # We'll mutate in place by updating properties; keep geometry as is
        updated_feat = {
            "type": "Feature",
            "properties": f.get("properties", {}),
            "geometry": f.get("geometry", {})
        }
        output_features.append(updated_feat)

    output = {
        "type": "FeatureCollection",
        "name": "output_gt",
        "crs": data.get("crs", {}),
        "features": output_features
    }

    # Write output
    with open(OUTPUT_GEOJSON_PATH, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2)

if __name__ == "__main__":
    main()