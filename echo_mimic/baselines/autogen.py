"""AutoGen baseline that mirrors the DSPy reference pipelines.

The implementation keeps the AutoGen flavour (planner → producer → critic),
but it now executes the same tasks as the DSPy baselines:
    * Stage-2 (local heuristics) for farm + energy domains
    * Stage-3 (global connectivity) for farm + energy domains
    * Stage-4 (nudging) for farm + energy domains

For each case we:
    1. load the relevant context (GeoJSON, prompts, personas, etc.),
    2. let a planner draft a short plan,
    3. ask a producer agent to emit Python code (or a policy message),
    4. run a critic pass; if it finds issues we request one revision,
    5. execute the produced artefact and score it against ground truth.

Scores, transcripts, and generated files are stored alongside the existing
DSPy outputs so that downstream analyses can compare both baselines.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from echo_mimic.baselines.dspy.energy_dataset import (
    format_agent_context,
    iter_stage_agents,
    load_cached_scenario,
)
from echo_mimic.common.models import build_model_client
from echo_mimic.config import Config
from echo_mimic.domains.energy_ev.evaluation import (
    evaluate_agent_nudge_response,
    evaluate_global_agent_policy_script,
    evaluate_local_agent_policy_script,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FarmLocalCase:
    farm_id: str
    farm_path: Path
    input_geojson_path: Path
    gt_path: Path
    prompt_text: str
    obs_text: str
    output_filename: str = "autogen_output.geojson"
    script_name: str = "autogen_local.py"


@dataclass
class FarmGlobalCase:
    farm_id: str
    conn_dir: Path
    input_geojson_path: Path
    gt_path: Path
    prompt_text: str
    obs_text: str
    output_filename: str = "autogen_output_directions.json"
    script_name: str = "autogen_global.py"


@dataclass
class FarmNudgeCase:
    farm_id: str
    nudge_dir: Path
    heur_dir: Path
    input_geojson_path: Path
    gt_dir_path: Path
    intens_code: str
    connect_code: str
    params_text: str
    farm_input_text: str
    output_filename: str = "autogen_output.geojson"
    script_name: str = "autogen_nudge.py"


@dataclass
class EnergyLocalCase:
    scenario_dir: Path
    agent_dir: Path
    agent_id: int
    prompt_text: str
    neighbor_text: str
    script_name: str = "autogen_local.py"


@dataclass
class EnergyGlobalCase:
    scenario_dir: Path
    agent_dir: Path
    agent_id: int
    prompt_text: str
    neighbor_text: str
    script_name: str = "autogen_global.py"


@dataclass
class EnergyNudgeCase:
    scenario_dir: Path
    agent_dir: Path
    agent_id: int
    persona_block: str
    recommended_block: str
    params_text: str
    prompt_text: str


# ---------------------------------------------------------------------------
# IO + scoring helpers (mirrors DSPy utilities)
# ---------------------------------------------------------------------------


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text()
    except Exception:
        return default


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_against_ground_truth(
    input_geojson: Dict[str, Any], pred_geojson: Dict[str, Any], gt_geojson: Dict[str, Any]
) -> float:
    """Same MAE-based metric used by Stage-2 evaluations."""
    pred_map: Dict[str, Tuple[float, float]] = {}
    for feature in pred_geojson.get("features", []):
        props = feature.get("properties", {})
        pid = str(props.get("id"))
        pred_map[pid] = (
            safe_float(props.get("margin_intervention"), 0.0),
            safe_float(props.get("habitat_conversion"), 0.0),
        )

    gt_map: Dict[str, Tuple[float, float]] = {}
    for feature in gt_geojson.get("features", []):
        props = feature.get("properties", {})
        pid = str(props.get("id"))
        gt_map[pid] = (
            safe_float(props.get("margin_intervention"), 0.0),
            safe_float(props.get("habitat_conversion"), 0.0),
        )

    margin_errs: List[float] = []
    habitat_errs: List[float] = []
    for feature in input_geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("type") == "hab_plots":
            continue
        pid = str(props.get("id"))
        gt_m, gt_h = gt_map.get(pid, (0.0, 0.0))
        pm, ph = pred_map.get(pid, (0.0, 0.0))
        margin_errs.append(abs(gt_m - pm))
        habitat_errs.append(abs(gt_h - ph))

    if not margin_errs:
        return 0.0
    mae_margin = sum(margin_errs) / len(margin_errs)
    mae_hab = sum(habitat_errs) / len(habitat_errs)
    return 1.0 / (mae_margin + mae_hab + 0.01)


ALLOWED_DIRS = {
    "north-east",
    "north-west",
    "south-east",
    "south-west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "ne",
    "nw",
    "se",
    "sw",
}

CANON_DIR = {
    "north-east": "north-east",
    "northeast": "north-east",
    "ne": "north-east",
    "north-west": "north-west",
    "northwest": "north-west",
    "nw": "north-west",
    "south-east": "south-east",
    "southeast": "south-east",
    "se": "south-east",
    "south-west": "south-west",
    "southwest": "south-west",
    "sw": "south-west",
}


def canon_dir_list(values: Iterable[Any]) -> List[str]:
    seen: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        key = value.strip().lower()
        if key in CANON_DIR:
            canon = CANON_DIR[key]
            if canon not in seen:
                seen.append(canon)
    return seen


def load_gt_direction_map(gt_any: Any) -> Dict[str, Tuple[List[str], List[str]]]:
    items: Sequence[Any]
    if isinstance(gt_any, dict) and "features" in gt_any:
        items = gt_any["features"]
    elif isinstance(gt_any, list):
        items = gt_any
    else:
        items = []
    output: Dict[str, Tuple[List[str], List[str]]] = {}
    for item in items:
        props = item.get("properties", item)
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)
        output[pid] = (
            canon_dir_list(props.get("margin_directions", [])),
            canon_dir_list(props.get("habitat_directions", [])),
        )
    return output


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def score_directions(
    input_geojson: Dict[str, Any], pred_any: Any, gt_any: Any
) -> float:
    """Per-plot Jaccard error (the Stage-3 metric)."""
    pred_map = load_gt_direction_map(pred_any)
    gt_map = load_gt_direction_map(gt_any)
    margin_errs: List[float] = []
    habitat_errs: List[float] = []
    for feature in input_geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("type") == "hab_plots":
            continue
        pid = str(props.get("id"))
        gt_m, gt_h = gt_map.get(pid, ([], []))
        pm, ph = pred_map.get(pid, ([], []))
        if pid not in pred_map:
            margin_errs.append(10.0)
            habitat_errs.append(10.0)
            continue
        margin_errs.append(1.0 - jaccard(pm, gt_m))
        habitat_errs.append(1.0 - jaccard(ph, gt_h))
    if not margin_errs:
        return 0.0
    mae_margin = sum(margin_errs) / len(margin_errs)
    mae_hab = sum(habitat_errs) / len(habitat_errs)
    return 1.0 / (mae_margin + mae_hab + 0.01)


def load_stage3_targets(gt_any: Any) -> Dict[str, Tuple[float, float]]:
    items: Sequence[Any]
    if isinstance(gt_any, dict) and "features" in gt_any:
        items = gt_any["features"]
    elif isinstance(gt_any, list):
        items = gt_any
    else:
        items = []
    targets: Dict[str, Tuple[float, float]] = {}
    for item in items:
        props = item.get("properties", item)
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)
        margin_fraction = min(1.0, len(canon_dir_list(props.get("margin_directions", []))) / 4.0)
        habitat_fraction = min(1.0, len(canon_dir_list(props.get("habitat_directions", []))) / 4.0)
        targets[pid] = (margin_fraction, habitat_fraction)
    return targets


def load_predicted_values(pred_any: Any) -> Dict[str, Tuple[float, float]]:
    if isinstance(pred_any, dict) and "features" in pred_any:
        items = pred_any["features"]
    elif isinstance(pred_any, list):
        items = pred_any
    else:
        items = []
    output: Dict[str, Tuple[float, float]] = {}
    for item in items:
        props = item.get("properties", item)
        pid = props.get("id", props.get("plot_id"))
        if pid is None:
            continue
        pid = str(pid)
        margin = safe_float(props.get("margin_intervention", 0.0))
        habitat = safe_float(props.get("habitat_conversion", 0.0))
        output[pid] = (max(0.0, min(1.0, margin)), max(0.0, min(1.0, habitat)))
    return output


def score_against_targets(
    input_geojson: Dict[str, Any],
    pred_any: Any,
    targets: Dict[str, Tuple[float, float]],
) -> float:
    pred_map = load_predicted_values(pred_any)
    margin_errs: List[float] = []
    habitat_errs: List[float] = []
    for feature in input_geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("type") == "hab_plots":
            continue
        pid = str(props.get("id"))
        target_margin, target_hab = targets.get(pid, (0.0, 0.0))
        if pid not in pred_map:
            margin_errs.append(10.0)
            habitat_errs.append(10.0)
            continue
        pm, ph = pred_map[pid]
        margin_errs.append(abs(target_margin - pm))
        habitat_errs.append(abs(target_hab - ph))
    if not margin_errs:
        return 0.0
    mae_margin = sum(margin_errs) / len(margin_errs)
    mae_hab = sum(habitat_errs) / len(habitat_errs)
    return 1.0 / (mae_margin + mae_hab + 0.01)


def run_python_code(
    code_text: str, cwd: Path, filename: str, timeout: int = 120
) -> Tuple[int, str, str, Path]:
    """Write code_text into cwd/filename and execute it."""
    cwd.mkdir(parents=True, exist_ok=True)
    script_path = cwd / filename
    script_path.write_text(code_text, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["python", script_path.name],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, script_path
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"Timeout after {timeout}s: {exc}", script_path
    except Exception as exc:
        return 1, "", f"Execution error: {exc}", script_path


# ---------------------------------------------------------------------------
# AutoGen baseline implementation
# ---------------------------------------------------------------------------


class AutoGenBaseline:
    """Coordinate planner/producer/critic loops that emit runnable artefacts."""

    def __init__(self, domain: str, log_dir: str = "outputs/autogen") -> None:
        self.domain = (domain or "farm").lower()
        self.log_dir = Path(log_dir)
        self.config = Config()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.model_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        mode: str,
        agent_id: Optional[str],
        model: Optional[str],
        data_hint: Optional[str],
    ) -> None:
        mode = (mode or "local").lower()
        self.model_name = model or os.getenv("GENAI_MODEL") or self.config.lm
        self.client = build_model_client(self.model_name)
        logger.info(
            "[AutoGen] domain=%s mode=%s model=%s agent=%s",
            self.domain,
            mode,
            self.model_name,
            agent_id,
        )
        root = self._resolve_data_root(data_hint)
        if self.domain == "farm":
            self._run_farm(mode, agent_id, root)
        elif self.domain in {"energy", "energy_ev"}:
            self._run_energy(mode, agent_id, root)
        else:
            raise ValueError(f"Unsupported AutoGen domain: {self.domain}")

    # ------------------------------------------------------------------
    # Farm handlers
    # ------------------------------------------------------------------

    def _run_farm(self, mode: str, agent_id: Optional[str], root: Path) -> None:
        if mode == "local":
            cases = self._load_farm_local_cases(root, agent_id)
            self._process_cases(cases, self._handle_farm_local_case, "farm-local")
        elif mode == "global":
            cases = self._load_farm_global_cases(root, agent_id)
            self._process_cases(cases, self._handle_farm_global_case, "farm-global")
        elif mode == "nudge":
            cases = self._load_farm_nudge_cases(root, agent_id)
            self._process_cases(cases, self._handle_farm_nudge_case, "farm-nudge")
        else:
            raise ValueError(f"Unsupported farm mode: {mode}")

    def _load_farm_local_cases(self, root: Path, agent_id: Optional[str]) -> List[FarmLocalCase]:
        pattern = f"farm_{agent_id}" if agent_id else "farm_*"
        cases: List[FarmLocalCase] = []
        for farm_path in sorted(root.glob(pattern)):
            input_geo = farm_path / "input.geojson"
            gt_path = farm_path / "output_gt.geojson"
            if not (input_geo.exists() and gt_path.exists()):
                continue
            case = FarmLocalCase(
                farm_id=farm_path.name,
                farm_path=farm_path,
                input_geojson_path=input_geo,
                gt_path=gt_path,
                prompt_text=read_text(farm_path / "prompt_input.txt", ""),
                obs_text=read_text(input_geo, ""),
            )
            cases.append(case)
        return cases

    def _load_farm_global_cases(self, root: Path, agent_id: Optional[str]) -> List[FarmGlobalCase]:
        pattern = f"farm_{agent_id}" if agent_id else "farm_*"
        cases: List[FarmGlobalCase] = []
        for farm_path in sorted(root.glob(pattern)):
            conn_dir = farm_path / "connectivity" / "run_1"
            input_geo = conn_dir / "input.geojson"
            gt_path = conn_dir / "output_gt_directions.json"
            if not (input_geo.exists() and gt_path.exists()):
                continue
            case = FarmGlobalCase(
                farm_id=farm_path.name,
                conn_dir=conn_dir,
                input_geojson_path=input_geo,
                gt_path=gt_path,
                prompt_text=read_text(conn_dir / "prompt_input.txt", ""),
                obs_text=read_text(input_geo, ""),
            )
            cases.append(case)
        return cases

    def _load_farm_nudge_cases(self, root: Path, agent_id: Optional[str]) -> List[FarmNudgeCase]:
        pattern = f"farm_{agent_id}" if agent_id else "farm_*"
        cases: List[FarmNudgeCase] = []
        for farm_path in sorted(root.glob(pattern)):
            nudge_dir = farm_path / "nudge"
            heur_dir = nudge_dir / "heuristics"
            intens_path = nudge_dir / "heuristics_gem_eco_intens.py"
            connect_path = nudge_dir / "heuristics_gem_eco_conn.py"
            gt_path = self._find_stage3_gt_path(farm_path)
            input_geo = farm_path / "input.geojson"
            if gt_path is None:
                continue
            if not (intens_path.exists() and connect_path.exists() and input_geo.exists()):
                continue
            heur_dir.mkdir(parents=True, exist_ok=True)
            params_text = (
                "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, "
                "'Canola/rapeseed': 1100, 'Barley': 120, 'Spring wheat': 200}, and costs (USD/ha): "
                "{'margin': {'implementation': 400, 'maintenance': 60}, "
                "'habitat': {'implementation': 300, 'maintenance': 70}, "
                "'agriculture': {'maintenance': 100}}."
            )
            case = FarmNudgeCase(
                farm_id=farm_path.name,
                nudge_dir=nudge_dir,
                heur_dir=heur_dir,
                input_geojson_path=input_geo,
                gt_dir_path=gt_path,
                intens_code=read_text(intens_path, ""),
                connect_code=read_text(connect_path, ""),
                params_text=params_text,
                farm_input_text=read_text(input_geo, ""),
            )
            cases.append(case)
        return cases

    def _find_stage3_gt_path(self, farm_path: Path) -> Optional[Path]:
        candidates = [
            farm_path / "connectivity" / "run_1" / "output_gt_directions.json",
            farm_path / "connectivity" / "output_gt_directions.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Energy handlers
    # ------------------------------------------------------------------

    def _run_energy(self, mode: str, agent_id: Optional[str], root: Path) -> None:
        mode = mode.lower()
        if mode == "local":
            cases = self._load_energy_local_cases(root, agent_id)
            self._process_cases(cases, self._handle_energy_local_case, "energy-local")
        elif mode == "global":
            cases = self._load_energy_global_cases(root, agent_id)
            self._process_cases(cases, self._handle_energy_global_case, "energy-global")
        elif mode == "nudge":
            cases = self._load_energy_nudge_cases(root, agent_id)
            self._process_cases(cases, self._handle_energy_nudge_case, "energy-nudge")
        else:
            raise ValueError(f"Unsupported energy mode: {mode}")

    def _load_energy_local_cases(self, root: Path, agent_id: Optional[str]) -> List[EnergyLocalCase]:
        cases: List[EnergyLocalCase] = []
        for scenario_dir, agent_dir, ag_id, _scenario, agent_cfg in iter_stage_agents(root, "local"):
            if agent_id is not None and str(ag_id) != str(agent_id):
                continue
            cases.append(
                EnergyLocalCase(
                    scenario_dir=scenario_dir,
                    agent_dir=agent_dir,
                    agent_id=ag_id,
                    prompt_text=read_text(agent_dir / "prompt_input.txt", ""),
                    neighbor_text=format_agent_context(agent_cfg),
                )
            )
        return cases

    def _load_energy_global_cases(self, root: Path, agent_id: Optional[str]) -> List[EnergyGlobalCase]:
        cases: List[EnergyGlobalCase] = []
        for scenario_dir, agent_dir, ag_id, _scenario, agent_cfg in iter_stage_agents(root, "global"):
            if agent_id is not None and str(ag_id) != str(agent_id):
                continue
            cases.append(
                EnergyGlobalCase(
                    scenario_dir=scenario_dir,
                    agent_dir=agent_dir,
                    agent_id=ag_id,
                    prompt_text=read_text(agent_dir / "prompt_input.txt", ""),
                    neighbor_text=format_agent_context(agent_cfg),
                )
            )
        return cases

    def _load_energy_nudge_cases(self, root: Path, agent_id: Optional[str]) -> List[EnergyNudgeCase]:
        cases: List[EnergyNudgeCase] = []
        for scenario_dir, agent_dir, ag_id, _scenario, agent_cfg in iter_stage_agents(root, "nudge"):
            if agent_id is not None and str(ag_id) != str(agent_id):
                continue
            context_path = agent_dir / "context.json"
            if not context_path.exists():
                continue
            context_payload = read_json(context_path)
            persona_block = json.dumps(
                {"persona": context_payload.get("persona", ""), "local_notes": context_payload.get("notes", "")},
                indent=2,
                sort_keys=True,
            )
            recommended_block = json.dumps(
                {
                    "recommended_usage": context_payload.get("recommended_usage", []),
                    "recommended_slots": context_payload.get("recommended_slots", []),
                },
                indent=2,
                sort_keys=True,
            )
            cases.append(
                EnergyNudgeCase(
                    scenario_dir=scenario_dir,
                    agent_dir=agent_dir,
                    agent_id=ag_id,
                    persona_block=persona_block,
                    recommended_block=recommended_block,
                    params_text=format_agent_context(agent_cfg),
                    prompt_text=read_text(agent_dir / "prompt_input.txt", ""),
                )
            )
        return cases

    # ------------------------------------------------------------------
    # Case processing helpers
    # ------------------------------------------------------------------

    def _process_cases(self, cases: Sequence[Any], handler, label: str) -> None:
        if not cases:
            logger.warning("[AutoGen] No %s cases available.", label)
            return
        scores: List[float] = []
        for case in cases:
            try:
                score = handler(case)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("[AutoGen] Error while processing %s: %s", label, exc)
                score = 0.0
            scores.append(score)
        mean_score = sum(scores) / max(len(scores), 1)
        logger.info("[AutoGen] %s cases=%d mean_score=%.4f", label, len(cases), mean_score)

    # ------------------------------------------------------------------
    # Farm case handlers
    # ------------------------------------------------------------------

    def _handle_farm_local_case(self, case: FarmLocalCase) -> float:
        context = self._truncate_context(case.obs_text, case.prompt_text)
        plan = self._planner_prompt(
            case_id=case.farm_id,
            role="FarmLocalPlanner",
            goal="design heuristics that compute margin_intervention and habitat_conversion for every ag plot",
            context=context,
        )
        code_text, critique = self._produce_code_with_critique(
            plan=plan,
            context=context,
            case_id=case.farm_id,
            constraints=textwrap.dedent(
                f"""
                Write a complete Python program that runs inside {case.farm_path}.
                Requirements:
                  - Load 'input.geojson' from the current directory.
                  - Only iterate over agricultural plots (properties.type == 'ag_plot').
                  - Produce margin_intervention and habitat_conversion values in [0,1].
                  - Write a FeatureCollection to '{case.output_filename}' where each feature
                    copies geometry + id from the input.
                  - No randomness, no external dependencies beyond Python stdlib.
                  - Do not print verbose logs; silence preferred.
                """
            ),
        )
        rc, stdout, stderr, script_path = run_python_code(
            code_text, case.farm_path, case.script_name, timeout=120
        )
        sections = [
            ("Planner plan", plan),
            ("Critique", critique),
            ("Python script", code_text),
            ("Execution stdout", stdout or "(empty)"),
            ("Execution stderr", stderr or "(empty)"),
        ]
        score = 0.0
        if rc == 0:
            score = self._score_farm_local(case)
        else:
            logger.warning(
                "[AutoGen][farm-local] %s script exited with rc=%s", case.farm_id, rc
            )
        self._write_transcript(case.farm_id, "farm-local", sections)
        logger.info("[AutoGen][farm-local] %s score=%.4f", case.farm_id, score)
        return score

    def _score_farm_local(self, case: FarmLocalCase) -> float:
        pred_path = case.farm_path / case.output_filename
        fallbacks = [
            case.farm_path / "dspy_output.geojson",
            case.farm_path / "output.geojson",
        ]
        for candidate in fallbacks:
            if pred_path.exists():
                break
            if candidate.exists():
                pred_path = candidate
        if not pred_path.exists():
            return 0.0
        try:
            pred = read_json(pred_path)
            input_geo = json.loads(case.obs_text) if case.obs_text else read_json(case.input_geojson_path)
            gt = read_json(case.gt_path)
        except Exception:
            return 0.0
        return score_against_ground_truth(input_geo, pred, gt)

    def _handle_farm_global_case(self, case: FarmGlobalCase) -> float:
        context = self._truncate_context(case.obs_text, case.prompt_text)
        plan = self._planner_prompt(
            case_id=case.farm_id,
            role="FarmGlobalPlanner",
            goal="predict habitat and margin direction lists for every agricultural plot",
            context=context,
        )
        code_text, critique = self._produce_code_with_critique(
            plan=plan,
            context=context,
            case_id=case.farm_id,
            constraints=textwrap.dedent(
                f"""
                Produce Python code that:
                  - Loads 'input.geojson' from {case.conn_dir}.
                  - For each ag plot choose margin_directions and habitat_directions using only:
                    north-east, north-west, south-east, south-west.
                  - Write the output FeatureCollection to '{case.output_filename}'.
                  - Keep logic deterministic and rely solely on fields inside input.geojson/prompt_input.txt.
                """
            ),
        )
        rc, stdout, stderr, script_path = run_python_code(
            code_text, case.conn_dir, case.script_name, timeout=120
        )
        sections = [
            ("Planner plan", plan),
            ("Critique", critique),
            ("Python script", code_text),
            ("Execution stdout", stdout or "(empty)"),
            ("Execution stderr", stderr or "(empty)"),
        ]
        score = 0.0
        if rc == 0:
            pred_path = case.conn_dir / case.output_filename
            if not pred_path.exists():
                alt = case.conn_dir / "output.json"
                if alt.exists():
                    pred_path = alt
            if pred_path.exists():
                try:
                    pred = read_json(pred_path)
                    input_geo = json.loads(case.obs_text) if case.obs_text else read_json(case.input_geojson_path)
                    gt = read_json(case.gt_path)
                    score = score_directions(input_geo, pred, gt)
                except Exception:
                    score = 0.0
        else:
            logger.warning("[AutoGen][farm-global] %s script exit rc=%s", case.farm_id, rc)
        self._write_transcript(case.farm_id, "farm-global", sections)
        logger.info("[AutoGen][farm-global] %s score=%.4f", case.farm_id, score)
        return score

    def _handle_farm_nudge_case(self, case: FarmNudgeCase) -> float:
        context_parts = [
            "Current heuristics:\n",
            case.intens_code[:2000],
            "\nDesired connectivity code:\n",
            case.connect_code[:2000],
        ]
        context = "".join(context_parts)
        plan = self._planner_prompt(
            case_id=case.farm_id,
            role="FarmNudgePlanner",
            goal="craft a \\communication{...} message that nudges the farmer toward the connectivity heuristics",
            context=context,
        )
        message, critique = self._produce_message(
            plan=plan,
            case_id=case.farm_id,
            instructions=textwrap.dedent(
                """
                Compose a single persuasive message formatted as \\communication{...}.
                The farmer follows Python code inside intens_code and may change it if
                you offer clear ecological or financial benefits. Do not enumerate plot IDs;
                focus on logic and incentives.
                """
            ),
            payload={
                "intens_code": case.intens_code,
                "connect_code": case.connect_code,
                "params": case.params_text,
            },
        )
        farmer_prompt = textwrap.dedent(
            f"""
            You are the farmer receiving that message. Update your heuristics by writing a Python script that:
              - loads 'input.geojson' from the current directory,
              - outputs a FeatureCollection to '{case.output_filename}' with id, margin_intervention, habitat_conversion,
              - only touches ag plots and keeps values in [0,1],
              - relies on deterministic logic and Python stdlib.
            """
        )
        context_text = textwrap.dedent(
            f"Message:\n{message}\n\nOriginal code:\n{case.intens_code}\n\nFarm input:\n{case.farm_input_text[:2000]}"
        )
        farmer_code, farmer_critique = self._produce_code_with_critique(
            plan="Use the message guidance to adjust the farmer's heuristics without breaking IO requirements.",
            context=context_text,
            case_id=f"{case.farm_id}-farmer",
            constraints=farmer_prompt,
        )

        # Prepare heuristics directory
        try:
            shutil.copyfile(case.input_geojson_path, case.heur_dir / "input.geojson")
        except Exception:
            pass

        rc, stdout, stderr, script_path = run_python_code(
            farmer_code, case.heur_dir, case.script_name, timeout=120
        )
        sections = [
            ("Planner plan", plan),
            ("Policy message", message),
            ("Message critique", critique),
            ("Farmer code critique", farmer_critique),
            ("Python script", farmer_code),
            ("Execution stdout", stdout or "(empty)"),
            ("Execution stderr", stderr or "(empty)"),
        ]
        score = 0.0
        if rc == 0:
            pred_path = case.heur_dir / case.output_filename
            if not pred_path.exists():
                alt = case.heur_dir / "output.geojson"
                if alt.exists():
                    pred_path = alt
            if pred_path.exists():
                try:
                    pred = read_json(pred_path)
                    input_geo = (
                        json.loads(case.farm_input_text)
                        if case.farm_input_text
                        else read_json(case.input_geojson_path)
                    )
                    targets = load_stage3_targets(read_json(case.gt_dir_path))
                    score = score_against_targets(input_geo, pred, targets)
                except Exception:
                    score = 0.0
        else:
            logger.warning("[AutoGen][farm-nudge] %s code exit rc=%s", case.farm_id, rc)
        self._write_transcript(case.farm_id, "farm-nudge", sections)
        logger.info("[AutoGen][farm-nudge] %s score=%.4f", case.farm_id, score)
        return score

    # ------------------------------------------------------------------
    # Energy case handlers
    # ------------------------------------------------------------------

    def _handle_energy_local_case(self, case: EnergyLocalCase) -> float:
        scenario_path = case.scenario_dir / "scenario.json"
        context = self._truncate_context(case.prompt_text, case.neighbor_text)
        plan = self._planner_prompt(
            case_id=f"{case.scenario_dir.name}-agent{case.agent_id}",
            role="EnergyLocalPlanner",
            goal="write local slot recommendations that respect feeder limits",
            context=context,
        )
        code_text, critique = self._produce_code_with_critique(
            plan=plan,
            context=context,
            case_id=str(case.agent_id),
            constraints=textwrap.dedent(
                """
                Emit a deterministic Python program that:
                  - loads 'scenario.json' from the current directory,
                  - reasons about the prompt text and neighbour context,
                  - writes a JSON list of seven integers (0-3) to 'local_policy_output.json'.
                """
            ),
        )
        score = 0.0
        try:
            scenario = load_cached_scenario(str(scenario_path))
            current_score, _ = evaluate_local_agent_policy_script(
                code_text,
                scenario=scenario,
                scenario_dir=case.agent_dir,
                agent_id=case.agent_id,
                output_filename="local_policy_output.json",
            )
            score = max(0.0, current_score)
        except Exception:
            score = 0.0
        sections = [
            ("Planner plan", plan),
            ("Critique", critique),
            ("Python script", code_text),
        ]
        self._write_transcript(f"{case.scenario_dir.name}_agent_{case.agent_id}", "energy-local", sections)
        logger.info("[AutoGen][energy-local] agent=%s score=%.4f", case.agent_id, score)
        return score

    def _handle_energy_global_case(self, case: EnergyGlobalCase) -> float:
        scenario_path = case.scenario_dir / "scenario.json"
        context = self._truncate_context(case.prompt_text, case.neighbor_text)
        plan = self._planner_prompt(
            case_id=f"{case.scenario_dir.name}-agent{case.agent_id}",
            role="EnergyGlobalPlanner",
            goal="produce a coordinated EV charging policy with seven slots",
            context=context,
        )
        code_text, critique = self._produce_code_with_critique(
            plan=plan,
            context=context,
            case_id=str(case.agent_id),
            constraints=textwrap.dedent(
                """
                Create a deterministic Python program that:
                  - reads 'scenario.json' from the current directory,
                  - outputs seven integers (0-3) in a JSON list saved to 'global_policy_output.json'.
                """
            ),
        )
        score = 0.0
        try:
            scenario = load_cached_scenario(str(scenario_path))
            current_score, _ = evaluate_global_agent_policy_script(
                code_text,
                scenario=scenario,
                scenario_dir=case.agent_dir,
                agent_id=case.agent_id,
                output_filename="global_policy_output.json",
            )
            score = max(0.0, current_score)
        except Exception:
            score = 0.0
        sections = [
            ("Planner plan", plan),
            ("Critique", critique),
            ("Python script", code_text),
        ]
        self._write_transcript(f"{case.scenario_dir.name}_agent_{case.agent_id}", "energy-global", sections)
        logger.info("[AutoGen][energy-global] agent=%s score=%.4f", case.agent_id, score)
        return score

    def _handle_energy_nudge_case(self, case: EnergyNudgeCase) -> float:
        context = textwrap.dedent(
            f"""
            Persona + notes:
            {case.persona_block}

            Global recommendations:
            {case.recommended_block}
            """
        )
        plan = self._planner_prompt(
            case_id=f"{case.scenario_dir.name}-agent{case.agent_id}",
            role="EnergyNudgePlanner",
            goal="produce a JSON nudge with persona, recommended_usage, and persuasive message",
            context=context,
        )
        message, critique = self._produce_message(
            plan=plan,
            case_id=str(case.agent_id),
            instructions=textwrap.dedent(
                """
                Respond with JSON only. Fields required:
                  - persona: which persona you are referencing (string).
                  - recommended_usage: seven usage vectors (four floats per day, values in [0, 1]) indicating how much to charge in each slot.
                  - message: persuasive reasoning referencing feeder + neighbour data.
                """
            ),
            payload={
                "persona_context": case.persona_block,
                "recommended_usage": case.recommended_block,
                "params": case.params_text,
                "prompt": case.prompt_text,
            },
        )
        score = 0.0
        try:
            scenario = load_cached_scenario(str(case.scenario_dir / "scenario.json"))
            current_score, _ = evaluate_agent_nudge_response(
                message,
                scenario=scenario,
                agent_id=case.agent_id,
            )
            score = max(0.0, current_score)
        except Exception:
            score = 0.0
        sections = [
            ("Planner plan", plan),
            ("Critique", critique),
            ("Message JSON", message),
        ]
        self._write_transcript(f"{case.scenario_dir.name}_agent_{case.agent_id}", "energy-nudge", sections)
        logger.info("[AutoGen][energy-nudge] agent=%s score=%.4f", case.agent_id, score)
        return score

    # ------------------------------------------------------------------
    # Prompting helpers
    # ------------------------------------------------------------------

    def _planner_prompt(self, case_id: str, role: str, goal: str, context: str) -> str:
        prompt = textwrap.dedent(
            f"""
            You are {role}, collaborating with teammates in an AutoGen loop.
            Goal: {goal}.
            Context (may be truncated):
            {context}

            Draft a numbered plan with 3-6 concrete steps.
            Each step should mention the file or field you need to inspect plus the intended output.
            """
        )
        return self._generate(prompt, tag=f"{case_id}-planner")

    def _produce_code_with_critique(
        self,
        *,
        plan: str,
        context: str,
        case_id: str,
        constraints: str,
    ) -> Tuple[str, str]:
        code_prompt = textwrap.dedent(
            f"""
            You are CoderAgent executing the plan below.
            Plan:
            {plan}

            Context excerpt:
            {context[:6000]}

            Constraints:
            {constraints}

            Emit ONLY runnable Python source code. Do not wrap it in Markdown fences.
            """
        )
        code_text = self._generate(code_prompt, tag=f"{case_id}-coder")
        critique_prompt = textwrap.dedent(
            f"""
            You are CriticAgent. Review this script for completeness and IO compliance.
            Script:
            {code_text}

            Respond with:
              VERDICT: PASS
            or
              VERDICT: FAIL - explanation
            If FAIL, be specific about missing files or mistakes.
            """
        )
        critique = self._generate(critique_prompt, tag=f"{case_id}-critic")
        if "VERDICT: PASS" not in critique.upper():
            revision_prompt = textwrap.dedent(
                f"""
                The critic found issues:
                {critique}

                Revise the script to address them. Output only the fixed Python code.
                """
            )
            code_text = self._generate(revision_prompt, tag=f"{case_id}-revise")
        return code_text, critique

    def _produce_message(
        self,
        *,
        plan: str,
        case_id: str,
        instructions: str,
        payload: Dict[str, str],
    ) -> Tuple[str, str]:
        message_prompt = textwrap.dedent(
            f"""
            You are PolicyMessenger. Plan:
            {plan}

            Instructions:
            {instructions}

            Payload:
            {json.dumps(payload, indent=2)[:8000]}

            Return only the final message.
            """
        )
        message = self._generate(message_prompt, tag=f"{case_id}-policy")
        critique_prompt = textwrap.dedent(
            f"""
            CriticAgent validating the policy message.
            Message:
            {message}

            Verify the formatting requirements and realism.
            Respond with VERDICT: PASS or VERDICT: FAIL - explanation.
            """
        )
        critique = self._generate(critique_prompt, tag=f"{case_id}-policy-critic")
        if "VERDICT: PASS" not in critique.upper():
            revision_prompt = textwrap.dedent(
                f"""
                Critique:
                {critique}

                Produce a revised message that fixes the issues. Keep the requested format.
                """
            )
            message = self._generate(revision_prompt, tag=f"{case_id}-policy-revise")
        return message, critique

    def _generate(self, prompt: str, tag: str) -> str:
        if self.client is None:
            raise RuntimeError("AutoGenBaseline client not configured")
        try:
            response = self.client.generate(prompt)
            return response.strip()
        except Exception as exc:
            logger.warning("[AutoGen] Generation failed (%s) for %s", exc, tag)
            return ""

    def _truncate_context(self, primary: str, secondary: str, limit: int = 6000) -> str:
        joined = f"{primary}\n\n{secondary}"
        if len(joined) <= limit:
            return joined
        return joined[:limit] + "\n...[truncated]..."

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _write_transcript(self, case_id: str, label: str, sections: Sequence[Tuple[str, str]]) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.log_dir / f"{self.domain}_{label}_{case_id}_{timestamp}.md"
        lines = [
            f"# AutoGen transcript ({self.domain} / {label})",
            f"Case: {case_id}",
            f"Model: {self.model_name}",
            "",
        ]
        for title, content in sections:
            lines.append(f"## {title}")
            lines.append((content or "").strip())
            lines.append("")
        write_text(path, "\n".join(lines))

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _resolve_data_root(self, data_hint: Optional[str]) -> Path:
        if self.domain == "farm":
            fallback = Path(self.config.farms_dir)
        else:
            fallback = Path(self.config.data_dir) / "energy_ev"
        if not data_hint:
            return fallback
        hint_path = Path(data_hint)
        if any(char in data_hint for char in "*?[]"):
            hint_path = hint_path.parent
        if hint_path.exists():
            return hint_path
        return fallback


__all__ = ["AutoGenBaseline"]
