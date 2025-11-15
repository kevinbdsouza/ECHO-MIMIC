"""Utility CLI for the carbon-aware EV charging ECHO-MIMIC pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from echo_mimic.domains.energy_ev import (
    build_stage_four_prompt,
    build_stage_one_prompt,
    build_stage_three_prompt,
    build_stage_two_prompt,
    dump_ground_truth,
    evaluate_global_policy_script,
    evaluate_local_policy_script,
    evaluate_nudge_response,
    load_scenario,
)


def _scenario_dir(path: str | None) -> Path:
    directory = Path(path or "data/energy_ev/scenario_1")
    if not directory.exists():
        raise FileNotFoundError(f"Scenario directory not found: {directory}")
    return directory


def cmd_generate(args: argparse.Namespace) -> None:
    scenario_dir = _scenario_dir(args.scenario_dir)
    scenario = load_scenario(scenario_dir / "scenario.json")

    dump_ground_truth(scenario_dir, scenario)

    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.joinpath("local").mkdir(exist_ok=True)
    scenario_dir.joinpath("global").mkdir(exist_ok=True)
    scenario_dir.joinpath("nudge").mkdir(exist_ok=True)

    scenario_dir.joinpath("prompt_input.txt").write_text(
        build_stage_one_prompt(scenario) + "\n", encoding="utf-8"
    )
    scenario_dir.joinpath("local", "prompt_input.txt").write_text(
        build_stage_two_prompt(scenario) + "\n", encoding="utf-8"
    )
    scenario_dir.joinpath("global", "prompt_input.txt").write_text(
        build_stage_three_prompt(scenario) + "\n", encoding="utf-8"
    )
    scenario_dir.joinpath("nudge", "prompt_input.txt").write_text(
        build_stage_four_prompt(scenario) + "\n", encoding="utf-8"
    )

    print(f"Updated prompts and ground truth for {scenario.scenario_id}")


def cmd_evaluate_local(args: argparse.Namespace) -> None:
    scenario_dir = _scenario_dir(args.scenario_dir)
    scenario = load_scenario(scenario_dir / "scenario.json")
    code = Path(args.script).read_text(encoding="utf-8")
    score, detail = evaluate_local_policy_script(
        code, scenario=scenario, scenario_dir=scenario_dir
    )
    print(json.dumps({"score": score, "detail": detail}, indent=2))


def cmd_evaluate_global(args: argparse.Namespace) -> None:
    scenario_dir = _scenario_dir(args.scenario_dir)
    scenario = load_scenario(scenario_dir / "scenario.json")
    code = Path(args.script).read_text(encoding="utf-8")
    score, detail = evaluate_global_policy_script(
        code, scenario=scenario, scenario_dir=scenario_dir
    )
    print(json.dumps({"score": score, "detail": detail}, indent=2))


def cmd_evaluate_nudge(args: argparse.Namespace) -> None:
    scenario_dir = _scenario_dir(args.scenario_dir)
    scenario = load_scenario(scenario_dir / "scenario.json")
    message = Path(args.message).read_text(encoding="utf-8")
    recommended = json.loads(
        scenario_dir.joinpath("nudge", "recommended_allocation.json").read_text(
            encoding="utf-8"
        )
    )["allocation"]
    score, detail = evaluate_nudge_response(
        message, scenario=scenario, recommended_allocation=recommended
    )
    print(json.dumps({"score": score, "detail": detail}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Helpers for the EV charging ECHO-MIMIC scenario",
    )
    parser.add_argument(
        "--scenario-dir",
        default="data/energy_ev/scenario_1",
        help="Path to the scenario directory (defaults to scenario_1)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Regenerate prompts and ground truth")
    gen.set_defaults(func=cmd_generate)

    eval_local = sub.add_parser("evaluate-local", help="Score a local heuristic script")
    eval_local.add_argument("script", help="Path to candidate Python script")
    eval_local.set_defaults(func=cmd_evaluate_local)

    eval_global = sub.add_parser("evaluate-global", help="Score a global heuristic script")
    eval_global.add_argument("script", help="Path to candidate Python script")
    eval_global.set_defaults(func=cmd_evaluate_global)

    eval_nudge = sub.add_parser("evaluate-nudge", help="Score a nudging message (JSON file)")
    eval_nudge.add_argument("message", help="Path to message JSON file")
    eval_nudge.set_defaults(func=cmd_evaluate_nudge)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
