"""Gemini-powered evolutionary search for local EV charging heuristics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from echo_mimic.common import build_model, configure_genai, ensure_rate_limiter
from echo_mimic.config import Config
from echo_mimic.domains.energy_ev import (
    build_stage_two_prompt,
    evaluate_local_agent_policy_script,
    load_scenario,
)

from energy_policy_evolution import EnergyPolicyContext, EnergyPolicyEvolutionRunner

cfg = Config()
rate_limiter = ensure_rate_limiter(cfg)

LOCAL_GUIDELINES = (
    "- Run entirely within the agent directory provided in the prompt.\n"
    "- Load scenario.json relative to the working directory to inspect neighbour exemplars.\n"
    "- Implement deterministic reasoning using comfort windows, feeder stress, and price cues.\n"
    "- Emit a 7-element list of slot indices (one per day) to local_policy_output.json via json.dump.\n"
    "- Do not read or write files outside the agent directory and avoid randomness.\n"
)


def init_models(model_name: str) -> tuple:
    configure_genai()
    system_instructions = (
        "You are an expert energy systems engineer and Python developer. "
        "Produce concise, fully runnable policy.py files that reason over the scenario context and output a 7-day slot plan."
    )
    fix_instructions = (
        "You repair Python scripts for EV charging heuristics. Given failing code and stderr, return a corrected policy.py."
    )
    heur_model = build_model(model_name, system_instructions, ensure_configured=False)
    fix_model = build_model(model_name, fix_instructions, ensure_configured=False)
    return heur_model, fix_model


def _ensure_prompt_file(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def run(
    *,
    scenario_dir: Path,
    agent_id: int,
    population_size: int,
    generations: int,
    inner_loop_size: int,
    seed: int,
    init: bool,
    model_name: str | None = None,
) -> None:
    scenario_path = scenario_dir / "scenario.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    scenario = load_scenario(scenario_path)
    if not any(agent.id == agent_id for agent in scenario.agents):
        raise ValueError(f"Unknown agent id {agent_id}")

    agent_dir = scenario_dir / "local" / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_stage_two_prompt(scenario, agent_id)
    _ensure_prompt_file(agent_dir / "prompt_input.txt", prompt)

    heur_model, fix_model = init_models(model_name or cfg.lm)

    def evaluator(code: str) -> tuple[float, dict[str, object]]:
        return evaluate_local_agent_policy_script(
            code,
            scenario=scenario,
            scenario_dir=agent_dir,
            agent_id=agent_id,
        )

    context = EnergyPolicyContext(
        stage_name="local",
        prompt=prompt,
        guidelines=LOCAL_GUIDELINES,
        agent_dir=agent_dir,
        evaluator=evaluator,
        heur_model=heur_model,
        fix_model=fix_model,
        rate_limiter=rate_limiter,
    )

    runner = EnergyPolicyEvolutionRunner(context, seed=seed)

    if init:
        runner.generate_initial_population(population_size, overwrite=True)

    result = runner.run(
        population_size=population_size,
        generations=generations,
        inner_loop_size=inner_loop_size,
    )

    best = result["best"]
    history = result["history"]

    agent_dir.joinpath("best_policy.py").write_text(best["code"], encoding="utf-8")
    agent_dir.joinpath("best_policy_detail.json").write_text(
        json.dumps(best.get("detail", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    agent_dir.joinpath("evo_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Completed evolution for agent {agent_id}: best score {best['score']:.3f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Gemini ES for a local EV agent.")
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("data/energy_ev/scenario_1"),
        help="Directory containing scenario.json and agent subfolders.",
    )
    parser.add_argument("--agent-id", type=int, required=True, help="Target agent id to optimise.")
    parser.add_argument("--population-size", type=int, default=8, help="Population size for the ES loop.")
    parser.add_argument("--generations", type=int, default=8, help="Number of generations to evolve.")
    parser.add_argument(
        "--inner-loop-size",
        type=int,
        default=16,
        help="Number of offspring attempts per generation.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="Skip regenerating the initial population with Gemini.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the Gemini model name used for generation and fixing.",
    )

    args = parser.parse_args(argv)

    run(
        scenario_dir=args.scenario_dir,
        agent_id=args.agent_id,
        population_size=max(1, args.population_size),
        generations=max(0, args.generations),
        inner_loop_size=max(1, args.inner_loop_size),
        seed=args.seed,
        init=not args.no_init,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
