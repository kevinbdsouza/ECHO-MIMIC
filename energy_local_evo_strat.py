"""Gemini-powered evolutionary search for local EV charging heuristics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import google.generativeai as genai

from echo_mimic.common import (
    CommandOutputCapture,
    build_model,
    configure_genai,
    ensure_rate_limiter,
    extract_python_code,
    make_code_validator,
)
from echo_mimic.config import Config
from echo_mimic.rate_limiter import send_message_with_retry
from echo_mimic.domains.energy_ev import (
    evaluate_local_agent_policy_script,
    load_scenario,
    build_stage_two_prompt,
)

from energy_evo_strat import EvolutionaryStrategy


cfg = Config()
rate_limiter = ensure_rate_limiter(cfg)


def _completion_text(completion: genai.types.GenerateContentResponse) -> str:
    if hasattr(completion, "text") and completion.text:
        return completion.text
    parts: List[str] = []
    for part in getattr(completion, "parts", []):
        text = getattr(part, "text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _json_dumps(payload: Dict[str, object]) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except TypeError:
        return str(payload)


def init_models(model_name: str) -> Tuple[genai.GenerativeModel, genai.GenerativeModel]:
    configure_genai()

    system_instructions = (
        "You are an expert energy systems engineer and Python developer. "
        "Write clean, well-structured Python scripts that agents can run locally. "
        "Scripts must read the EV scenario description from paths relative to the agent directory, "
        "apply clear heuristics using neighbour exemplars, and write the selected slot index to a JSON file. "
        "Return only the full contents of policy.py without commentary."
    )

    fix_instructions = (
        "You repair Python scripts for EV charging heuristics. "
        "Given faulty code and stderr output, return a corrected policy.py implementation that executes without errors."
    )

    heur_model = build_model(model_name, system_instructions, ensure_configured=False)
    fix_model = build_model(model_name, fix_instructions, ensure_configured=False)
    return heur_model, fix_model


def _make_validator(workdir: Path, capture: CommandOutputCapture, fix_model: genai.GenerativeModel):
    return make_code_validator(
        workdir=workdir,
        capture=capture,
        fix_model=fix_model,
        rate_limiter=rate_limiter,
        default_script="policy.py",
        default_attempts=2,
    )


def _ensure_prompt_file(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def _generate_policy_code(
    *,
    base_prompt: str,
    heur_model: genai.GenerativeModel,
    validator,
    retry: int = 3,
) -> str:
    prompt = (
        base_prompt
        + "\n\nRespond with the executable Python source for policy.py. "
        "Do not include explanations, markdown fences, or placeholder text."
    )

    last_error: Exception | None = None
    for _ in range(retry):
        try:
            session = heur_model.start_chat(history=[])
            completion = send_message_with_retry(session, prompt, rate_limiter)
            response_text = _completion_text(completion)
            code = extract_python_code(response_text) or response_text
            code = code.strip()
            if not code:
                raise ValueError("Gemini returned an empty response")
            return validator(code, script_name="policy.py", max_attempts=2)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    raise RuntimeError(f"Failed to generate policy code: {last_error}")


def _mutate_policy_code(
    *,
    base_prompt: str,
    current_code: str,
    evaluation_detail: Dict[str, object] | None,
    heur_model: genai.GenerativeModel,
    validator,
    rng: random.Random,
) -> str:
    focus_hints = [
        "adjust how price sensitivity is weighted",
        "emphasise neighbour exemplar alignment",
        "tune comfort penalty handling",
        "consider fallback logic for ties",
    ]
    hint = rng.choice(focus_hints)
    feedback = _json_dumps(evaluation_detail or {"note": "initial mutation"})

    prompt = (
        base_prompt
        + "\n\nFeedback from the latest evaluation:\n"
        + feedback
        + "\n\nThe current policy.py implementation is:\n```python\n"
        + current_code
        + "\n```\n\nApply a subtle, meaningful change that "
        + hint
        + ". Return the full, updated policy.py with no commentary."
    )

    session = heur_model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response_text = _completion_text(completion)
    code = extract_python_code(response_text) or response_text
    code = code.strip()
    if not code:
        raise RuntimeError("Mutation produced an empty response")
    return validator(code, script_name="policy.py", max_attempts=2)


def _load_population(population_dir: Path, limit: int) -> List[str]:
    payloads: List[str] = []
    for path in sorted(population_dir.glob("*.py"))[:limit]:
        payloads.append(path.read_text(encoding="utf-8"))
    return payloads


def _write_population(population_dir: Path, codes: Sequence[str]) -> None:
    population_dir.mkdir(parents=True, exist_ok=True)
    for idx, code in enumerate(codes, start=1):
        path = population_dir / f"candidate_{idx:02d}.py"
        path.write_text(code, encoding="utf-8")


def run(
    *,
    scenario_dir: Path,
    agent_id: int,
    population_size: int,
    generations: int,
    seed: int,
    init: bool,
    model_name: str | None = None,
) -> None:
    scenario_path = scenario_dir / "scenario.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    scenario = load_scenario(scenario_path)
    try:
        agent = next(agent for agent in scenario.agents if agent.id == agent_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown agent id {agent_id}") from exc

    agent_dir = scenario_dir / "local" / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    population_dir = agent_dir / "population"

    prompt = build_stage_two_prompt(scenario, agent_id)
    _ensure_prompt_file(agent_dir / "prompt_input.txt", prompt)

    capture = CommandOutputCapture()
    heur_model, fix_model = init_models(model_name or cfg.lm)
    validator = _make_validator(agent_dir, capture, fix_model)

    if init:
        seeds: List[str] = []
        for _ in range(population_size):
            code = _generate_policy_code(
                base_prompt=prompt,
                heur_model=heur_model,
                validator=validator,
            )
            seeds.append(code)
        _write_population(population_dir, seeds)

    initial_population = _load_population(population_dir, population_size)
    if not initial_population:
        raise RuntimeError("No initial population found. Run with --init to generate seeds.")

    feedback_map: Dict[str, Dict[str, object]] = {}

    def evaluate(code: str) -> Tuple[float, Dict[str, object]]:
        score, detail = evaluate_local_agent_policy_script(
            code,
            scenario=scenario,
            scenario_dir=agent_dir,
        )
        feedback_map[code] = detail
        return score, detail

    def mutate(code: str, rng: random.Random) -> str:
        return _mutate_policy_code(
            base_prompt=prompt,
            current_code=code,
            evaluation_detail=feedback_map.get(code),
            heur_model=heur_model,
            validator=validator,
            rng=rng,
        )

    strategy = EvolutionaryStrategy(
        evaluate=evaluate,
        mutate=mutate,
        population_size=population_size,
        seed=seed,
    )

    best, history = strategy.run(initial_population, generations)

    agent_dir.joinpath("best_policy.py").write_text(best.payload, encoding="utf-8")
    agent_dir.joinpath("best_policy_detail.json").write_text(
        json.dumps(best.detail, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    agent_dir.joinpath("evo_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Completed evolution for agent {agent_id}: best score {best.score:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Gemini ES for a local EV agent.")
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("data/energy_ev/scenario_1"),
        help="Directory containing scenario.json and agent subfolders.",
    )
    parser.add_argument("--agent-id", type=int, required=True, help="Target agent id to optimise.")
    parser.add_argument("--population-size", type=int, default=6, help="Population size for the ES loop.")
    parser.add_argument("--generations", type=int, default=8, help="Number of generations to evolve.")
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
        seed=args.seed,
        init=not args.no_init,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

