"""Gemini-powered evolutionary search for EV behaviour nudges."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import google.generativeai as genai

from echo_mimic.common import build_model, configure_genai, ensure_rate_limiter
from echo_mimic.config import Config
from echo_mimic.rate_limiter import send_message_with_retry
from echo_mimic.domains.energy_ev import (
    build_stage_four_prompt,
    evaluate_agent_nudge_response,
    load_scenario,
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


def _normalise_message(raw_text: str) -> str:
    payload = json.loads(raw_text)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def init_model(model_name: str) -> genai.GenerativeModel:
    configure_genai()
    system_instructions = (
        "You craft persuasive yet respectful energy-behaviour nudges. "
        "Given persona context and policy code, output a JSON object with keys persona, "
        "recommended_slot, and message. Keep the JSON compact and factual."
    )
    return build_model(model_name, system_instructions, ensure_configured=False)


def _ensure_prompt_file(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def _generate_message(
    *,
    base_prompt: str,
    heur_model: genai.GenerativeModel,
    retry: int = 3,
) -> str:
    prompt = (
        base_prompt
        + "\n\nRespond with a single JSON object string matching the specification. "
        "Do not include extra prose."
    )

    last_error: Exception | None = None
    for _ in range(retry):
        try:
            session = heur_model.start_chat(history=[])
            completion = send_message_with_retry(session, prompt, rate_limiter)
            response_text = _completion_text(completion).strip()
            if not response_text:
                raise ValueError("Gemini returned an empty response")
            return _normalise_message(response_text)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    raise RuntimeError(f"Failed to generate nudge message: {last_error}")


def _mutate_message(
    *,
    base_prompt: str,
    current_message: str,
    evaluation_detail: Dict[str, object] | None,
    heur_model: genai.GenerativeModel,
    rng: random.Random,
) -> str:
    focus_hints = [
        "strengthen the carbon benefit framing",
        "acknowledge the household's comfort preferences",
        "connect the request explicitly to neighbour examples",
        "offer a concrete incentive or reassurance",
    ]
    hint = rng.choice(focus_hints)
    feedback = json.dumps(evaluation_detail or {"note": "initial mutation"}, indent=2, sort_keys=True)

    pretty_message = json.dumps(json.loads(current_message), indent=2, sort_keys=True)

    prompt = (
        base_prompt
        + "\n\nLatest evaluation feedback:\n"
        + feedback
        + "\n\nCurrent JSON message:\n"
        + pretty_message
        + "\n\nRevise the message to "
        + hint
        + ". Respond with a compact JSON object string only."
    )

    session = heur_model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response_text = _completion_text(completion).strip()
    if not response_text:
        raise RuntimeError("Mutation produced an empty response")
    return _normalise_message(response_text)


def _load_population(population_dir: Path, limit: int) -> List[str]:
    payloads: List[str] = []
    for path in sorted(population_dir.glob("*.json"))[:limit]:
        payloads.append(path.read_text(encoding="utf-8").strip())
    return payloads


def _write_population(population_dir: Path, messages: Sequence[str]) -> None:
    population_dir.mkdir(parents=True, exist_ok=True)
    for idx, message in enumerate(messages, start=1):
        path = population_dir / f"candidate_{idx:02d}.json"
        path.write_text(message + "\n", encoding="utf-8")


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
    if not any(agent.id == agent_id for agent in scenario.agents):
        raise ValueError(f"Unknown agent id {agent_id}")

    agent_dir = scenario_dir / "nudge" / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    population_dir = agent_dir / "population"

    prompt = build_stage_four_prompt(scenario, agent_id, scenario_dir=scenario_dir)
    _ensure_prompt_file(agent_dir / "prompt_input.txt", prompt)

    heur_model = init_model(model_name or cfg.lm)

    if init:
        messages: List[str] = []
        for _ in range(population_size):
            message = _generate_message(
                base_prompt=prompt,
                heur_model=heur_model,
            )
            messages.append(message)
        _write_population(population_dir, messages)

    initial_population = _load_population(population_dir, population_size)
    if not initial_population:
        raise RuntimeError("No initial population found. Run with --init to generate seeds.")

    feedback_map: Dict[str, Dict[str, object]] = {}

    def evaluate(message: str) -> tuple[float, Dict[str, object]]:
        score, detail = evaluate_agent_nudge_response(
            message,
            scenario=scenario,
            agent_id=agent_id,
        )
        feedback_map[message] = detail
        return score, detail

    def mutate(message: str, rng: random.Random) -> str:
        return _mutate_message(
            base_prompt=prompt,
            current_message=message,
            evaluation_detail=feedback_map.get(message),
            heur_model=heur_model,
            rng=rng,
        )

    strategy = EvolutionaryStrategy(
        evaluate=evaluate,
        mutate=mutate,
        population_size=population_size,
        seed=seed,
    )

    best, history = strategy.run(initial_population, generations)

    agent_dir.joinpath("best_message.json").write_text(best.payload + "\n", encoding="utf-8")
    agent_dir.joinpath("best_message_detail.json").write_text(
        json.dumps(best.detail, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    agent_dir.joinpath("evo_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Completed nudge evolution for agent {agent_id}: best score {best.score:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Gemini ES for EV nudges.")
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("data/energy_ev/scenario_1"),
        help="Directory containing scenario.json and agent subfolders.",
    )
    parser.add_argument("--agent-id", type=int, required=True, help="Target agent id to optimise.")
    parser.add_argument("--population-size", type=int, default=6, help="Population size for the ES loop.")
    parser.add_argument("--generations", type=int, default=6, help="Number of generations to evolve.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="Skip regenerating the initial population with Gemini.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the Gemini model name used for nudge generation.",
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

