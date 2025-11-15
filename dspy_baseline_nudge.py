# dspy_nudge_baseline.py
# DSPy baseline for Stage-4 (nudging / MiMiC):
# - A Policy LLM proposes a message \communication{...}
# - A Farmer LLM (fixed prompt) turns that message into a COMPLETE Python program
#   that reads input.geojson and writes output.geojson with margin_intervention
#   and habitat_conversion for every ag_plot.
# - We execute that program and score it against targets derived from Stage-3
#   direction ground-truth (fraction = |dirs| / 4). Fitness = 1 / (MAE_m + MAE_h + 0.01)
#
# Notes:
# - No demos; instruction-only MIPROv2 (0-shot) for the Policy message.
# - Farmer LLM is a fixed dspy.Predict used inside the metric (not optimized).
# - Paths match your repo layout; robust fallbacks included.

import os, json, glob, random, math, subprocess, pathlib, shutil
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot
from echo_mimic.dspy_rate_limiter import configure_dspy_with_rate_limiting

from echo_mimic.config import Config


# --------------------------
# Basic IO helpers
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
        json.dump(data, f, indent=2)


# --------------------------
# Executing generated code
# --------------------------

def run_python_code_in_cwd(code_text: str, cwd: str, filename: str = "tmp_dspynudge.py",
                           timeout_sec: int = 120) -> Tuple[int, str, str, str]:
    """
    Writes code_text to cwd/filename and executes: python filename
    Returns (returncode, stdout, stderr, script_path)
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
# Targets from Stage-3 directions
# --------------------------

DIR_KEYS = {"north-east","north-west","south-east","south-west",
            "northeast","northwest","southeast","southwest","ne","nw","se","sw"}
DIR_CANON = {
    "north-east":"north-east", "northeast":"north-east","ne":"north-east",
    "north-west":"north-west","northwest":"north-west","nw":"north-west",
    "south-east":"south-east","southeast":"south-east","se":"south-east",
    "south-west":"south-west","southwest":"south-west","sw":"south-west",
}

def canon_dir_list(xs: Any) -> List[str]:
    out: List[str] = []
    if isinstance(xs, (list, tuple)):
        for v in xs:
            if isinstance(v, str):
                s = v.strip().lower()
                if s in DIR_CANON:
                    c = DIR_CANON[s]
                    if c not in out:
                        out.append(c)
    return out

def load_stage3_gt_directions(gt_any: Any) -> Dict[str, Tuple[float, float]]:
    """
    Accepts Stage-3 GT in either list-of-dicts or FeatureCollection with properties.
    Returns numeric targets per plot id:
        targets[pid] = (margin_fraction, habitat_fraction)
    where fraction = len(directions)/4.0 (capped 0..1).
    """
    items: List[Dict[str, Any]]
    if isinstance(gt_any, list):
        items = gt_any
    elif isinstance(gt_any, dict) and "features" in gt_any:
        items = gt_any["features"]
    else:
        items = []

    out: Dict[str, Tuple[float, float]] = {}
    for it in items:
        props = it.get("properties", it)
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)
        m_dirs = canon_dir_list(props.get("margin_directions", []))
        h_dirs = canon_dir_list(props.get("habitat_directions", []))
        m = max(0.0, min(1.0, len(m_dirs) / 4.0))
        h = max(0.0, min(1.0, len(h_dirs) / 4.0))
        out[pid] = (m, h)
    return out


# --------------------------
# Read predicted output.geojson (program output)
# --------------------------

def load_predicted_values(pred_any: Any) -> Dict[str, Tuple[float, float]]:
    """
    Accepts model output as FeatureCollection or list of dicts with properties:
      id, margin_intervention, habitat_conversion
    Returns: {pid: (margin, habitat)}
    """
    if isinstance(pred_any, dict) and "features" in pred_any:
        items = pred_any["features"]
    elif isinstance(pred_any, list):
        items = pred_any
    else:
        items = []

    out: Dict[str, Tuple[float, float]] = {}
    for it in items:
        props = it.get("properties", it)
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)
        try:
            m = float(props.get("margin_intervention", 0.0))
        except Exception:
            m = 0.0
        try:
            h = float(props.get("habitat_conversion", 0.0))
        except Exception:
            h = 0.0
        m = max(0.0, min(1.0, m))
        h = max(0.0, min(1.0, h))
        out[pid] = (m, h)
    return out


# --------------------------
# Fitness: 1 / (MAE_margin + MAE_hab + 0.01)
# Missing plot -> penalty 10 (per channel), matching your behavior
# --------------------------

def score_against_targets(input_geojson: Dict[str, Any],
                          pred_any: Any,
                          targets: Dict[str, Tuple[float, float]]) -> float:
    pred_map = load_predicted_values(pred_any)

    margin_errs: List[float] = []
    habitat_errs: List[float] = []

    for feat in input_geojson.get("features", []):
        props = feat.get("properties", {})
        if props.get("type") == "hab_plots":
            continue
        pid = str(props.get("id"))

        tm, th = targets.get(pid, (0.0, 0.0))
        if pid not in pred_map:
            margin_errs.append(10.0)
            habitat_errs.append(10.0)
            continue
        pm, ph = pred_map[pid]
        margin_errs.append(abs(tm - pm))
        habitat_errs.append(abs(th - ph))

    if not margin_errs:
        return 0.0

    mae_m = sum(margin_errs) / len(margin_errs)
    mae_h = sum(habitat_errs) / len(habitat_errs)
    total = mae_m + mae_h + 0.01
    return 1.0 / total


# --------------------------
# DSPy Signatures / Modules
# --------------------------

class PolicyMessageSig(dspy.Signature):
    """
    You are a policy expert in land-use, incentives, and communication.
    TASK: Write ONE persuasive message that nudges a farmer from the
          current ecological-intensification heuristics (Python code)
          toward ecological-connectivity heuristics (Python code).
    FORMAT: Return the final message as \\communication{...}
    CONTEXT YOU RECEIVE:
      - intens_code: Python code the farmer currently follows
      - connect_code: Python code representing the desired global/connectivity logic
      - params: Price/cost parameters string
      - farm_input_json: Raw input.geojson (for high-level context)
    RULES:
      - Don't include code in the message; just the message.
      - Incentives may adjust costs/compensation; be specific but plausible.
      - Do not mention plot IDs; the farmer will map logic, not IDs.
    """
    intens_code: str = dspy.InputField(desc="Farmer's current Python heuristics.")
    connect_code: str = dspy.InputField(desc="Connectivity-target heuristics (desired direction).")
    params: str = dspy.InputField(desc="Price/cost parameters text.")
    farm_input_json: str = dspy.InputField(desc="Raw contents of input.geojson for the farm.")
    message: str = dspy.OutputField(desc="Final message embedded as \\communication{...}")

class FarmerEditSig(dspy.Signature):
    """
    You are the farmer. You currently follow Python heuristics `intens_code`.
    You receive a policy message (in \\communication{...}) and may update your code
    if the message aligns with your interests/incentives.

    TASK: Output a COMPLETE Python program (plain text, no backticks) that:
      1) loads 'input.geojson' from CWD
      2) writes 'output.geojson' with a FeatureCollection containing one feature per
         ag_plot with properties {id, margin_intervention, habitat_conversion} in [0,1].
      3) Keep IO structure the same; only adjust the decision logic per message.
      4) Deterministic, no randomness. No extra files.

    Use only stdlib (json, math). Do NOT invent new output property names.
    """
    intens_code: str = dspy.InputField(desc="Current Python heuristics the farmer uses.")
    message: str = dspy.InputField(desc="Policy message in \\communication{...} format.")
    params: str = dspy.InputField(desc="Price/cost parameters text (if you choose to use).")
    farm_input_json: str = dspy.InputField(desc="Raw contents of input.geojson (context only).")
    python_code: str = dspy.OutputField(desc="A complete, runnable Python script as plain text.")


class PolicyNudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.policy = dspy.Predict(PolicyMessageSig)

    def forward(self, intens_code: str, connect_code: str, params: str, farm_input_json: str):
        out = self.policy(intens_code=intens_code, connect_code=connect_code,
                          params=params, farm_input_json=farm_input_json)
        return dspy.Prediction(message=out.message)


# Fixed farmer (not optimized) used inside metric
FARMER = None
def get_farmer_predictor():
    global FARMER
    if FARMER is None:
        FARMER = dspy.Predict(FarmerEditSig)
    return FARMER


# --------------------------
# Dataset plumbing
# --------------------------

def find_stage3_gt_path(farm_path: str) -> Optional[str]:
    c1 = os.path.join(farm_path, "connectivity", "run_1", "output_gt_directions.json")
    c2 = os.path.join(farm_path, "connectivity", "output_gt_directions.json")
    return c1 if os.path.exists(c1) else (c2 if os.path.exists(c2) else None)

def load_nudge_farms(farms_dir: str) -> List[dspy.Example]:
    """
    For each farm_X:
      - input.geojson   (from farm root)
      - nudge/heuristics working dir
      - nudge/heuristics_gem_eco_intens.py (current)
      - nudge/heuristics_gem_eco_conn.py   (desired)
      - connectivity/*/output_gt_directions.json (targets)
    """
    examples: List[dspy.Example] = []
    for farm_path in sorted(glob.glob(os.path.join(farms_dir, "farm_*"))):
        input_geo = os.path.join(farm_path, "input.geojson")
        nudge_dir = os.path.join(farm_path, "nudge")
        heur_dir = os.path.join(nudge_dir, "heuristics")
        os.makedirs(heur_dir, exist_ok=True)

        intens_path = os.path.join(nudge_dir, "heuristics_gem_eco_intens.py")
        connect_path = os.path.join(nudge_dir, "heuristics_gem_eco_conn.py")
        gt_path = find_stage3_gt_path(farm_path)

        if not (os.path.exists(input_geo) and os.path.exists(intens_path) and os.path.exists(connect_path) and gt_path):
            continue

        example = dspy.Example(
            intens_code=read_text(intens_path, ""),
            connect_code=read_text(connect_path, ""),
            params=("These are the crop prices in USD/Tonne: "
                    "{'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
                    "'Barley': 120, 'Spring wheat': 200}, and costs (USD/ha): "
                    "{'margin': {'implementation': 400,  'maintenance': 60}, "
                    "'habitat': {'implementation': 300, 'maintenance': 70}, "
                    "'agriculture': {'maintenance': 100}}."),
            farm_input_json=read_text(input_geo, "")
        ).with_inputs("intens_code", "connect_code", "params", "farm_input_json")

        example.farm = {
            "farm_path": farm_path,
            "input_geojson_path": input_geo,
            "nudge_dir": nudge_dir,
            "heur_dir": heur_dir,
            "gt_dir_path": gt_path,
            "tmp_code_name": "tmp_dspynudge.py",
            "best_policy_path": os.path.join(nudge_dir, "best_policy_dspy.txt"),
            "best_python_path": os.path.join(nudge_dir, "best_python_dspy.py"),
        }
        examples.append(example)
    return examples

def filter_with_gt(examples: List[dspy.Example]) -> List[dspy.Example]:
    return [ex for ex in examples if os.path.exists(ex.farm["gt_dir_path"])]


# --------------------------
# Metric wrapper (runs Farmer LLM + executes code)
# --------------------------

BEST: Dict[str, Tuple[float, str, str]] = {}  # nudge_dir -> (best_score, best_message, best_code)

def metric_nudge_wrapper(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    1) Take policy message from pred.message
    2) Call Farmer LLM (fixed) to produce a COMPLETE Python script
    3) Ensure heur_dir has input.geojson; run the script; expect output.geojson
    4) Score against Stage-3 targets (fractions from directions)
    5) Keep best message/code snapshots
    """
    farm = example.farm
    nudge_dir = farm["nudge_dir"]
    heur_dir = farm["heur_dir"]

    # Ensure a fresh input.geojson in heur_dir (script reads CWD/input.geojson)
    try:
        shutil.copyfile(farm["input_geojson_path"], os.path.join(heur_dir, "input.geojson"))
    except Exception:
        pass

    # Load GT directions and convert to numeric targets
    gt_any = read_json(farm["gt_dir_path"])
    targets = load_stage3_gt_directions(gt_any)

    # Call Farmer LLM to translate message into Python program
    msg = pred.message or ""
    farmer = get_farmer_predictor()
    farm_prog = farmer(intens_code=example.intens_code,
                       message=msg,
                       params=example.params,
                       farm_input_json=example.farm_input_json)
    code_text = farm_prog.python_code or ""

    # Execute the produced code in heur_dir
    rc, out, err, script_path = run_python_code_in_cwd(code_text, heur_dir, filename=farm["tmp_code_name"])
    if rc != 0:
        return 0.0

    # Read output (accept exact 'output.geojson'; tolerate 'dspy_output.geojson' as fallback)
    pred_path = os.path.join(heur_dir, "output.geojson")
    if not os.path.exists(pred_path):
        alt = os.path.join(heur_dir, "dspy_output.geojson")
        if os.path.exists(alt):
            pred_path = alt
        else:
            return 0.0

    pred_any = read_json(pred_path)

    # Load farm input (for list of plots)
    input_geojson = json.loads(example.farm_input_json) if example.farm_input_json else read_json(farm["input_geojson_path"])

    score = score_against_targets(input_geojson, pred_any, targets)

    # Track best artifacts
    prev = BEST.get(nudge_dir, (-math.inf, "", ""))
    if score > prev[0]:
        BEST[nudge_dir] = (score, msg, code_text)
        with open(farm["best_policy_path"], "w") as f:
            f.write(msg)
        with open(farm["best_python_path"], "w") as f:
            f.write(code_text)

    return score


# --------------------------
# Main
# --------------------------

def main():
    load_dotenv()
    cfg = Config()

    # Configure LM with rate limiting (single LM used by both policy + farmer)
    configure_dspy_with_rate_limiting(model=cfg.lm, seed=cfg.seed)

    farms_dir = getattr(cfg, "farms_dir", os.path.join(cfg.data_dir, "crop_inventory", "syn_farms"))
    all_examples = load_nudge_farms(farms_dir)
    if not all_examples:
        raise SystemExit(f"No nudge-ready farms found under: {farms_dir}")

    trainset = filter_with_gt(all_examples)
    evalset = trainset
    print(f"[Stage-4 DSPy] farms with GT: {len(trainset)}")

    # Optimizer for the POLICY only (Farmer is fixed)
    use_bootstrap = getattr(cfg, "bootstrap_demos", 0) and getattr(cfg, "bootstrap_demos", 0) > 0
    trials = getattr(cfg, "trials", 50)
    auto_level = getattr(cfg, "auto", "heavy")  # light|medium|heavy

    tele = MIPROv2(metric=metric_nudge_wrapper, auto=auto_level, seed=cfg.seed, verbose=True)
    compiled = tele.compile(
        PolicyNudge(),
        trainset=trainset,
        valset=evalset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0
    )
    mode_desc = f"MIPROv2(auto={auto_level}, trials={trials})"

    # Evaluate
    scores: List[float] = []
    per_farm: List[Tuple[str, float]] = []
    for ex in evalset:
        pred = compiled(intens_code=ex.intens_code,
                        connect_code=ex.connect_code,
                        params=ex.params,
                        farm_input_json=ex.farm_input_json)
        s = metric_nudge_wrapper(ex, pred, trace=None)
        farm_id = os.path.basename(ex.farm["farm_path"])
        per_farm.append((f"{farm_id}", s))
        scores.append(s)

    mean_score = sum(scores) / max(1, len(scores))
    print(f"[DSPy-Nudge 0-shot] optimizer={mode_desc} | ALL_n={len(scores)} | mean_score={mean_score:.4f}")
    for farm_id, s in per_farm:
        print(f"  - {farm_id}: {s:.4f}")

    # Save compiled artifact
    out_dir = os.path.join(farms_dir, "dspy")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "dspy_nudge_0shot_program.json")

    # scrub demos for purist 0-shot artifact
    if hasattr(compiled, "policy") and hasattr(compiled.policy, "demos"):
        try:
            compiled.policy.demos = []
        except Exception:
            pass

    compiled.save(out_json)
    print(f"Saved optimized policy program to {out_json}")


if __name__ == "__main__":
    main()
