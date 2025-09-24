# dspy_rule_baseline.py  (code-producing baseline)
import os, sys, json, glob, random, pathlib, math, copy, subprocess, tempfile
import math, time
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from config import Config
import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot
from dotenv import load_dotenv
from dspy_rate_limiter import configure_dspy_with_rate_limiting

# top-level
BEST = {}  # farm_path -> (best_score, code_text)

# --------------------------
# Utility: file IO
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
    # ensure parent dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

# --------------------------
# Evaluation metric (your MAE-based scheme)
# --------------------------

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def score_against_ground_truth(input_geojson: Dict[str, Any],
                               pred_geojson: Dict[str, Any],
                               gt_geojson: Dict[str, Any]) -> float:
    """
    Mirrors your evaluate_heuristics(): compute mean absolute error for margin + habitat, across ag plots.
    MAE = mean(|pred - gt|) for each property; Total = MAE_margin + MAE_habitat
    Metric returned = 1 / (Total + 0.01). Higher is better.
    """
    input_feats = input_geojson.get("features", [])
    pred_map = {}
    for pf in pred_geojson.get("features", []):
        pid = str(pf.get("properties", {}).get("id"))
        pm = safe_float(pf.get("properties", {}).get("margin_intervention"), 0.0)
        ph = safe_float(pf.get("properties", {}).get("habitat_conversion"), 0.0)
        pred_map[pid] = (pm, ph)

    gt_map = {}
    for gf in gt_geojson.get("features", []):
        gid = str(gf.get("properties", {}).get("id"))
        gm = safe_float(gf.get("properties", {}).get("margin_intervention"), 0.0)
        gh = safe_float(gf.get("properties", {}).get("habitat_conversion"), 0.0)
        gt_map[gid] = (gm, gh)

    margin_errs: List[float] = []
    habitat_errs: List[float] = []

    for feat in input_geojson.get("features", []):
        props = feat.get("properties", {})
        if props.get("type") == "hab_plots":
            continue
        pid = str(props.get("id"))
        gt_m, gt_h = gt_map.get(pid, (0.0, 0.0))
        pm, ph = pred_map.get(pid, (0.0, 0.0))
        margin_errs.append(abs(gt_m - pm))
        habitat_errs.append(abs(gt_h - ph))

    if not margin_errs:
        return 0.0
    total_margin_err = sum(margin_errs) / len(margin_errs)
    total_hab_err = sum(habitat_errs) / len(habitat_errs)
    total_err = total_margin_err + total_hab_err + 0.01
    return 1.0 / total_err

# --------------------------
# DSPy: code-producing program
# --------------------------

class GenerateHeuristic(dspy.Signature):
    """Write a COMPLETE Python program that:
    (1) loads 'input.geojson' from the current working directory,
    (2) decides per-plot interventions only for agricultural plots (properties.type == "ag_plot"),
    (3) writes a FeatureCollection to 'dspy_output.geojson' with one feature per ag plot,
        copying the geometry from input and setting properties:
          - id (same as input)
          - margin_intervention: float in [0,1]
          - habitat_conversion:  float in [0,1]
    Constraints and guidance:
      - Consider ONLY what is visible in 'input.geojson'. Do not use or reference any ground truth.
      - Do NOT create or write any other files. Write exactly one file: 'dspy_output.geojson'.
      - Ensure EVERY ag plot receives a prediction; if uncertain, output 0.0.
      - Keep values in [0,1]. Use simple, deterministic logic (no randomness).
      - Use only Python standard library plus 'json' and 'math'. Avoid third-party imports.
      - Do not print verbose logs; silent execution preferred.
    Return ONLY the Python code, no backticks or extra text.
    """
    obs_json: str = dspy.InputField(desc="Raw contents of this farm's input.geojson (for context only).")
    neighbor_icl: str = dspy.InputField(desc="Optional neighbor context text (may be empty).")
    python_code: str = dspy.OutputField(desc="A complete, runnable Python script as plain text.")

class DSPyCode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(GenerateHeuristic)
    def forward(self, obs_json: str, neighbor_icl: str = ""):
        out = self.gen(obs_json=obs_json, neighbor_icl=neighbor_icl)
        return dspy.Prediction(python_code=out.python_code)

# --------------------------
# Data plumbing (train==eval==all farms with GT)
# --------------------------

def load_unlabeled_farms(farms_dir: str) -> List[dspy.Example]:
    examples = []
    for farm_path in sorted(glob.glob(os.path.join(farms_dir, "farm_*"))):
        input_geo = os.path.join(farm_path, "input.geojson")
        if not os.path.exists(input_geo):
            continue
        obs_json = read_text(input_geo, default="")
        nb_icl = read_text(os.path.join(farm_path, "prompt_input.txt"), default="")

        ex = dspy.Example(obs_json=obs_json, neighbor_icl=nb_icl).with_inputs("obs_json", "neighbor_icl")
        ex.farm = {
            "farm_path": farm_path,
            "input_geojson_path": input_geo,
            "gt_path": os.path.join(farm_path, "output_gt.geojson"),
            "pred_path": os.path.join(farm_path, "dspy_output.geojson"),
        }
        examples.append(ex)
    return examples

def filter_with_gt(examples: List[dspy.Example]) -> List[dspy.Example]:
    return [ex for ex in examples if os.path.exists(ex.farm["gt_path"])]

# --------------------------
# Runner: execute generated code inside the farm folder
# --------------------------

def run_generated_code(code_text: str, farm_path: str, timeout_sec: int = 90) -> Tuple[int, str, str]:
    """
    Write 'tmp_dspyrule.py' to the farm folder and execute it with the current Python.
    Returns (returncode, stdout, stderr). We don't delete the script so runs are reproducible.
    """
    script_path = os.path.join(farm_path, "tmp_dspyrule.py")
    try:
        with open(script_path, "w") as f:
            f.write(code_text)
    except Exception as e:
        return (1, "", f"WRITE_ERROR: {e}")

    try:
        proc = subprocess.run(
            [sys.executable, os.path.basename(script_path)],
            cwd=farm_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True
        )
        return (proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired:
        return (1, "", "TIMEOUT")
    except Exception as e:
        return (1, "", f"RUNTIME_ERROR: {e}")

# --------------------------
# Metric for DSPy (code → run → read output → score)
# --------------------------

def metric_code_wrapper(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    1) Take generated python_code.
    2) Save & run it in the farm folder (cwd = farm_path).
    3) The code must write 'dspy_output.geojson'.
    4) Score that file against output_gt.geojson with your MAE metric.
    """
    farm_path = example.farm["farm_path"]
    input_geojson = json.loads(example.obs_json) if example.obs_json else read_json(example.farm["input_geojson_path"])
    gt_path = example.farm["gt_path"]
    pred_path = example.farm["pred_path"]

    if not os.path.exists(gt_path):
        return 0.0

    code_text = pred.python_code.strip() if getattr(pred, "python_code", None) else ""
    if not code_text:
        return 0.0

    # Run generated script
    rc, out, err = run_generated_code(code_text, farm_path=farm_path, timeout_sec=90)
    if rc != 0:
        # Failed run → zero reward
        return 0.0

    # Check that the expected output was produced
    if not os.path.exists(pred_path):
        # Some models may write a different name; try to rescue by checking common mistakes
        alt = os.path.join(farm_path, "output.geojson")
        if os.path.exists(alt):
            # Move it to the expected name for consistency
            try:
                os.replace(alt, pred_path)
            except Exception:
                pass

    if not os.path.exists(pred_path):
        return 0.0

    # Load and score
    try:
        pred_geojson = read_json(pred_path)
    except Exception:
        return 0.0

    try:
        gt_geojson = read_json(gt_path)
    except Exception:
        return 0.0

    score = score_against_ground_truth(input_geojson, pred_geojson, gt_geojson)
    
    # Track and save best performing code for each farm
    if rc == 0 and os.path.exists(pred_path):
        prev = BEST.get(farm_path, (-math.inf, ""))
        if score > prev[0]:
            BEST[farm_path] = (score, code_text)
            with open(os.path.join(farm_path, "best_dspyrule.py"), "w") as f:
                f.write(code_text)
    
    return score

# --------------------------
# Main
# --------------------------

def main():
    # Load environment variables from .env file
    load_dotenv()

    cfg = Config()
    # Ensure save dir for compiled program
    save_dir = os.path.join(cfg.farms_dir, "dspy")
    os.makedirs(save_dir, exist_ok=True)

    # Configure LM with rate limiting (LiteLLM under the hood)
    configure_dspy_with_rate_limiting(model=cfg.lm, seed=cfg.seed)

    # Build dataset: ALL farms with GT (train == eval)
    all_examples = load_unlabeled_farms(cfg.farms_dir)
    if not all_examples:
        raise SystemExit(f"No farms found under: {cfg.farms_dir}")
    trainset = filter_with_gt(all_examples)
    evalset = trainset
    if not trainset:
        raise SystemExit("No farms with output_gt.geojson present; nothing to optimize/evaluate against.")

    # Choose optimizer
    tele = MIPROv2(metric=metric_code_wrapper, auto=cfg.auto, seed=cfg.seed, verbose=True)
    compiled = tele.compile(
        DSPyCode(),
        trainset=trainset,
        valset=evalset,                 # fine to set equal since you're training on all
        max_bootstrapped_demos=0,       # strict 0-shot
        max_labeled_demos=0
    )
    mode_desc = f"MIPROv2(auto={cfg.auto}, trials={getattr(cfg, 'trials', None)})"

    # Evaluate on the same set (by design)
    scores: List[float] = []
    per_farm: List[Tuple[str, float]] = []
    for ex in evalset:
        pred = compiled(obs_json=ex.obs_json, neighbor_icl=ex.neighbor_icl)
        s = metric_code_wrapper(ex, pred, trace=None)
        farm_id = os.path.basename(ex.farm["farm_path"])
        per_farm.append((farm_id, s))
        scores.append(s)

    mean_score = sum(scores) / max(1, len(scores))
    print(f"[DSPy-Code 0-shot] optimizer={mode_desc} | ALL_n={len(scores)} | mean_score={mean_score:.4f}")
    for farm_id, s in per_farm:
        print(f"  - {farm_id}: {s:.4f}")

    # Save compiled program (optimized instruction/program text)
    out_json = os.path.join(save_dir, "dspy_code_0shot_program.json")
    compiled.save(out_json)
    print(f"Saved optimized program to {out_json}")

if __name__ == "__main__":
    main()