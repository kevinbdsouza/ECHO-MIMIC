# dspy_directions_baseline.py
# DSPy baseline for ECHO Stage 3 (global connectivity directions)
# - No demos; instruction-only optimization with MIPROv2 (0-shot)
# - Code-producing: LM must output a COMPLETE Python script we run per farm
# - Metric: per-plot Jaccard error over margin_directions and habitat_directions,
#           aggregated as mean + epsilon, then fitness = 1 / (total_error)

import os, json, glob, random, math, copy, subprocess, tempfile, time, pathlib
from typing import Dict, Any, List, Tuple, Set, Optional
from dotenv import load_dotenv

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot
from dspy_rate_limiter import configure_dspy_with_rate_limiting

# Your existing config module (same as Stage-2 baseline)
from config import Config


# --------------------------
# IO helpers
# --------------------------

def read_text(path: str, default: str = "") -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return default


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# --------------------------
# Code execution helper
# --------------------------

def run_python_code_in_cwd(code_text: str, cwd: str, filename: str = "tmp_dspydirs.py",
                           timeout_sec: int = 120) -> Tuple[int, str, str, str]:
    """
    Writes `code_text` to `cwd/filename`, executes it with `python filename`, and returns:
    (returncode, stdout, stderr, script_path)
    """
    os.makedirs(cwd, exist_ok=True)
    script_path = os.path.join(cwd, filename)
    with open(script_path, "w") as f:
        f.write(code_text)

    try:
        proc = subprocess.run(
            ["python", filename],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        return proc.returncode, proc.stdout, proc.stderr, script_path
    except subprocess.TimeoutExpired as te:
        return 124, "", f"Timeout after {timeout_sec}s\n{te}", script_path
    except Exception as e:
        return 1, "", f"Execution error: {e}", script_path


# --------------------------
# Normalization helpers (directions, structures)
# --------------------------

ALLOWED_DIRS = {
    "north-east", "north-west", "south-east", "south-west",
    "northeast", "northwest", "southeast", "southwest",
    "ne", "nw", "se", "sw"
}

CANON = {
    "ne": "north-east", "northeast": "north-east", "north-east": "north-east",
    "nw": "north-west", "northwest": "north-west", "north-west": "north-west",
    "se": "south-east", "southeast": "south-east", "south-east": "south-east",
    "sw": "south-west", "southwest": "south-west", "south-west": "south-west",
}


def canon_dir(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    s = name.strip().lower()
    if s in CANON:
        return CANON[s]
    return None


def canon_dir_list(xs: Any) -> List[str]:
    out: List[str] = []
    if isinstance(xs, (list, tuple)):
        for v in xs:
            c = canon_dir(v)
            if c and c not in out:
                out.append(c)
    return out


# --------------------------
# Ground-truth + predicted parsing
# --------------------------

def load_gt_direction_map(gt_any: Any) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """
    Accepts ground truth in either list-of-dicts or FeatureCollection-like form.
    Returns: {plot_id: (set(margin_dirs), set(hab_dirs))}
    """
    out: Dict[str, Tuple[Set[str], Set[str]]] = {}

    # GT often comes as a list of dicts per plot
    if isinstance(gt_any, list):
        it = gt_any
    elif isinstance(gt_any, dict) and "features" in gt_any:
        it = gt_any["features"]
    else:
        it = []

    for item in it:
        props = item.get("properties", item) if isinstance(item, dict) else {}
        # ID could be under 'id' or 'plot_id'
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)

        gt_m = canon_dir_list(props.get("margin_directions", []))
        gt_h = canon_dir_list(props.get("habitat_directions", []))

        out[pid] = (set(gt_m), set(gt_h))

    return out


def load_pred_direction_map(pred_any: Any) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """
    Accepts predicted output that might be:
      - FeatureCollection with features->properties (id, margin_directions, habitat_directions), or
      - List of dicts with those keys.
    Returns: {plot_id: (set(margin_dirs), set(hab_dirs))}
    """
    out: Dict[str, Tuple[Set[str], Set[str]]] = {}
    if isinstance(pred_any, dict) and "features" in pred_any:
        it = pred_any["features"]
    elif isinstance(pred_any, list):
        it = pred_any
    else:
        it = []

    for item in it:
        props = item.get("properties", item) if isinstance(item, dict) else {}
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)

        pm = canon_dir_list(props.get("margin_directions", []))
        ph = canon_dir_list(props.get("habitat_directions", []))
        out[pid] = (set(pm), set(ph))

    return out


# --------------------------
# Metric = 1 / (MAE_margin + MAE_hab + 0.01),
# where per-plot error = 1 - Jaccard(set_pred, set_gt)
# --------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    u = a | b
    if not u:
        return 0.0  # matches your code: union empty -> jaccard 0
    return len(a & b) / len(u)


def score_directions(input_geojson: Dict[str, Any],
                     pred_any: Any,
                     gt_any: Any) -> float:
    gt_map = load_gt_direction_map(gt_any)
    pred_map = load_pred_direction_map(pred_any)

    margin_errs: List[float] = []
    habitat_errs: List[float] = []

    for feat in input_geojson.get("features", []):
        props = feat.get("properties", {})
        if props.get("type") == "hab_plots":
            continue
        pid = str(props.get("id"))

        gt_m, gt_h = gt_map.get(pid, (set(), set()))
        pm, ph = pred_map.get(pid, (set(), set()))

        # If id missing entirely in prediction, penalize harshly (as in your code)
        if pid not in pred_map:
            margin_errs.append(10.0)
            habitat_errs.append(10.0)
            continue

        margin_errs.append(1.0 - jaccard(pm, gt_m))
        habitat_errs.append(1.0 - jaccard(ph, gt_h))

    if not margin_errs:
        return 0.0

    mae_m = sum(margin_errs) / len(margin_errs)
    mae_h = sum(habitat_errs) / len(habitat_errs)
    total = mae_m + mae_h + 0.01
    return 1.0 / total


# --------------------------
# DSPy Signature & Module (code-producing)
# --------------------------

class GenerateDirectionsCode(dspy.Signature):
    """
    Produce a COMPLETE Python program that:
      1) loads 'input.geojson' from CWD
      2) for each ag plot (properties.type == 'ag_plot'), decides
         margin_directions and habitat_directions as lists of strings chosen ONLY from:
         ['north-east','north-west','south-east','south-west']
      3) writes a FeatureCollection to 'dspy_output_directions.json' with one feature per ag plot:
         properties: { id, margin_directions, habitat_directions } and geometry copied from input

    Constraints:
      - Use ONLY information visible in 'input.geojson' (e.g., label, yield, nbs, geometry).
      - DO NOT copy/peek at any ground truth. DO NOT assign by plot_id rules.
      - Every ag plot must be present.
      - Keep outputs deterministic; no randomness.
      - Use only Python stdlib (json, math, collections, etc.)
      - Write exactly one file named 'dspy_output_directions.json'.
    """
    obs_json: str = dspy.InputField(desc="Raw contents of this farm's connectivity/input.geojson.")
    neighbor_icl: str = dspy.InputField(desc="Neighbor examples / ICL context (prompt_input.txt), may be empty.")
    python_code: str = dspy.OutputField(desc="A complete, runnable Python script as plain text (no backticks).")


class DSPyCodeDirections(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(GenerateDirectionsCode)

    def forward(self, obs_json: str, neighbor_icl: str = ""):
        out = self.gen(obs_json=obs_json, neighbor_icl=neighbor_icl)
        return dspy.Prediction(python_code=out.python_code)


# --------------------------
# Dataset plumbing (Stage-3 connectivity folders)
# --------------------------

def load_connectivity_farms(farms_dir: str) -> List[dspy.Example]:
    """
    Scans farms/*/connectivity for input+gt.
    Example metadata carries paths needed by the metric.
    """
    examples: List[dspy.Example] = []
    for farm_path in sorted(glob.glob(os.path.join(farms_dir, "farm_*"))):
        conn_dir = os.path.join(farm_path, "connectivity", "run_1")
        input_geo = os.path.join(conn_dir, "input.geojson")
        if not os.path.exists(input_geo):
            continue

        ex = dspy.Example(
            obs_json=read_text(input_geo, default=""),
            neighbor_icl=read_text(os.path.join(conn_dir, "prompt_input.txt"), default="")
        ).with_inputs("obs_json", "neighbor_icl")

        ex.farm = {
            "farm_path": farm_path,
            "conn_dir": conn_dir,
            "input_geojson_path": input_geo,
            "gt_path": os.path.join(conn_dir, "output_gt_directions.json"),
            "pred_path": os.path.join(conn_dir, "dspy_output_directions.json"),
            "tmp_code_name": "tmp_dspydirs.py",
        }
        examples.append(ex)
    return examples


def filter_with_gt(examples: List[dspy.Example]) -> List[dspy.Example]:
    return [ex for ex in examples if os.path.exists(ex.farm["gt_path"])]


# --------------------------
# Metric wrapper used by DSPy (runs the emitted code)
# --------------------------

BEST: Dict[str, Tuple[float, str]] = {}  # farm_path -> (best_score, code_text)


def metric_code_wrapper(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    1) Write the emitted Python code to connectivity/tmp_dspydirs.py
    2) Run it; the script must write dspy_output_directions.json
    3) Load that file and compute Stage-3 score vs output_gt_directions.json
    4) Track and save the best code as best_dspydirs.py
    """
    farm = example.farm
    conn_dir = farm["conn_dir"]
    input_geojson = json.loads(example.obs_json) if example.obs_json else read_json(farm["input_geojson_path"])
    gt_path = farm["gt_path"]

    # Require GT
    if not os.path.exists(gt_path):
        return 0.0
    gt_any = read_json(gt_path)

    code_text = pred.python_code or ""
    # Run generated code
    rc, out, err, script_path = run_python_code_in_cwd(code_text, conn_dir, filename=farm["tmp_code_name"])
    if rc != 0:
        # If code failed, no score
        # (Optionally could attempt to parse partial outputs, but we keep strict)
        return 0.0

    # Read predicted output (accept either our target name or a fallback if author wrote 'output.json')
    pred_path = farm["pred_path"]
    if not os.path.exists(pred_path):
        alt = os.path.join(conn_dir, "output.json")
        if os.path.exists(alt):
            pred_path = alt
        else:
            return 0.0

    pred_any = read_json(pred_path)
    score = score_directions(input_geojson, pred_any, gt_any)

    # Track best
    prev = BEST.get(conn_dir, (-math.inf, ""))
    if score > prev[0]:
        BEST[conn_dir] = (score, code_text)
        with open(os.path.join(conn_dir, "best_dspydirs.py"), "w") as f:
            f.write(code_text)

    return score


# --------------------------
# Main
# --------------------------

def main():
    load_dotenv()
    cfg = Config()

    # LM selection with rate limiting (same pattern as your Stage-2 baseline)
    # - cfg.lm: provider/model name (e.g., "openai/gpt-4o" or your local proxy)
    # - cfg.seed: reproducibility
    # - optional cfg.trials for compile()
    configure_dspy_with_rate_limiting(model=cfg.lm, seed=cfg.seed)

    # Build dataset from connectivity folders
    farms_dir = getattr(cfg, "farms_dir", os.path.join(cfg.data_dir, "crop_inventory", "syn_farms"))
    all_examples = load_connectivity_farms(farms_dir)
    if not all_examples:
        raise SystemExit(f"No connectivity farms found under: {farms_dir}")

    trainset = filter_with_gt(all_examples)
    evalset = trainset
    print(f"[Stage-3 DSPy] farms with GT: {len(trainset)}")

    # Choose optimizer: default = MIPROv2, 0-shot
    use_bootstrap = getattr(cfg, "bootstrap_demos", 0) and getattr(cfg, "bootstrap_demos", 0) > 0
    trials = getattr(cfg, "trials", 50)
    auto_level = getattr(cfg, "auto", "heavy")  # "light" | "medium" | "heavy"

    tele = MIPROv2(metric=metric_code_wrapper, auto=auto_level, seed=cfg.seed, verbose=True)
    compiled = tele.compile(
        DSPyCodeDirections(),
        trainset=trainset,
        valset=evalset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0
    )
    mode_desc = f"MIPROv2(auto={auto_level}, trials={trials})"

    # Evaluate compiled program
    scores: List[float] = []
    per_farm: List[Tuple[str, float]] = []
    for ex in evalset:
        pred = compiled(obs_json=ex.obs_json, neighbor_icl=ex.neighbor_icl)
        s = metric_code_wrapper(ex, pred, trace=None)
        farm_id = os.path.basename(ex.farm["farm_path"])
        per_farm.append((farm_id, s))
        scores.append(s)

    mean_score = sum(scores) / max(1, len(scores))
    print(f"[DSPy-Directions 0-shot] optimizer={mode_desc} | ALL_n={len(scores)} | mean_score={mean_score:.4f}")
    for farm_id, s in per_farm:
        print(f"  - {farm_id}: {s:.4f}")

    # Save optimized instructions/program for exact reproduction
    out_dir = os.path.join(farms_dir, "dspy")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "dspy_directions_0shot_program.json")

    # (Optional) scrub cached exemplars for a purist “0-demo” artifact
    if hasattr(compiled, "gen") and hasattr(compiled.gen, "demos"):
        try:
            compiled.gen.demos = []
        except Exception:
            pass

    compiled.save(out_json)
    print(f"Saved optimized program to {out_json}")


if __name__ == "__main__":
    main()
