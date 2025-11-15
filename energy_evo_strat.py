"""Evolutionary strategy utilities for EV charging heuristics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from echo_mimic.domains.energy_ev import (
    EVScenario,
    evaluate_agent_nudge_response,
    evaluate_global_agent_policy_script,
    evaluate_local_agent_policy_script,
    load_scenario,
)


@dataclass
class Candidate:
    """Single candidate heuristic or message."""

    payload: str
    score: float = float("-inf")
    detail: Dict[str, object] = field(default_factory=dict)

    def clone(self) -> "Candidate":
        return Candidate(self.payload, self.score, dict(self.detail))


class EvolutionaryStrategy:
    """Lightweight evolutionary loop for per-agent optimisation."""

    def __init__(
        self,
        *,
        evaluate: Callable[[str], Tuple[float, Dict[str, object]]],
        mutate: Callable[[str, random.Random], str],
        population_size: int,
        seed: int,
    ) -> None:
        self._evaluate = evaluate
        self._mutate = mutate
        self._population_size = max(1, population_size)
        self._rng = random.Random(seed)
        self._cache: Dict[str, Tuple[float, Dict[str, object]]] = {}

    def run(
        self,
        initial_payloads: Sequence[str],
        generations: int,
    ) -> Tuple[Candidate, List[Dict[str, object]]]:
        if not initial_payloads:
            raise ValueError("initial population must not be empty")

        population = [Candidate(payload) for payload in initial_payloads[: self._population_size]]
        for candidate in population:
            candidate.score, candidate.detail = self._cached_evaluate(candidate.payload)

        history: List[Dict[str, object]] = [self._snapshot(0, population)]
        best = max(population, key=lambda cand: cand.score)

        for generation in range(1, generations + 1):
            survivors = self._select(population)
            next_population: List[Candidate] = [candidate.clone() for candidate in survivors]

            while len(next_population) < self._population_size:
                parent = self._rng.choice(survivors)
                mutated_payload = self._mutate(parent.payload, self._rng)
                next_population.append(Candidate(mutated_payload))

            for candidate in next_population[len(survivors) :]:
                candidate.score, candidate.detail = self._cached_evaluate(candidate.payload)

            population = next_population
            history.append(self._snapshot(generation, population))
            if population:
                best = max([best] + list(population), key=lambda cand: cand.score)

        return best, history

    def _cached_evaluate(self, payload: str) -> Tuple[float, Dict[str, object]]:
        cached = self._cache.get(payload)
        if cached is not None:
            return cached[0], dict(cached[1])

        score, detail = self._evaluate(payload)
        self._cache[payload] = (score, dict(detail))
        return score, dict(detail)

    def _select(self, population: Sequence[Candidate]) -> List[Candidate]:
        if not population:
            raise ValueError("population is empty")
        elite_count = max(1, len(population) // 2)
        ordered = sorted(population, key=lambda cand: cand.score, reverse=True)
        return ordered[:elite_count]

    def _snapshot(self, generation: int, population: Sequence[Candidate]) -> Dict[str, object]:
        scores = [candidate.score for candidate in population]
        best_candidate = max(population, key=lambda cand: cand.score) if population else None
        return {
            "generation": generation,
            "population_size": len(population),
            "best_score": best_candidate.score if best_candidate else None,
            "mean_score": statistics.fmean(scores) if scores else None,
            "best_preview": (best_candidate.payload[:120] if best_candidate else None),
            "best_detail": best_candidate.detail if best_candidate else None,
        }


def _load_payloads_from_directory(stage: str, directory: Path) -> List[str]:
    if not directory.exists():
        raise FileNotFoundError(f"Seed directory not found: {directory}")

    payloads: List[str] = []
    if stage in {"local", "global"}:
        for path in sorted(directory.glob("*.py")):
            payloads.append(path.read_text(encoding="utf-8"))
    elif stage == "nudge":
        for path in sorted(directory.glob("*.json")):
            payloads.append(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.txt")):
            payloads.append(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if not payloads:
        raise ValueError(f"No seed files discovered in {directory}")
    return payloads


def _mutate_python_code(code: str, rng: random.Random) -> str:
    import re

    float_matches = list(re.finditer(r"(?<![\w.])(\d+\.\d+)", code))
    if not float_matches:
        return code

    match = rng.choice(float_matches)
    value = float(match.group(1))
    delta = rng.uniform(-0.2, 0.2)
    new_value = max(0.0, value + delta)
    start, end = match.span(1)
    return code[:start] + f"{new_value:.3f}" + code[end:]


def _mutate_nudge_message(message: str, rng: random.Random) -> str:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return message

    phrases = [
        "This keeps the transformer happy.",
        "It also trims carbon peaks.",
        "Neighbors already lean this way.",
        "You'll still respect your comfort window.",
    ]
    base_message = str(payload.get("message", ""))
    addition = rng.choice(phrases)
    payload["message"] = (base_message.strip() + " " + addition).strip()
    return json.dumps(payload, indent=2)


def _make_evaluator(
    *,
    stage: str,
    scenario: EVScenario,
    scenario_dir: Path,
    agent_id: int,
) -> Callable[[str], Tuple[float, Dict[str, object]]]:
    if stage == "local":
        return lambda code: evaluate_local_agent_policy_script(
            code,
            scenario=scenario,
            scenario_dir=scenario_dir,
            agent_id=agent_id,
        )
    if stage == "global":
        return lambda code: evaluate_global_agent_policy_script(
            code,
            scenario=scenario,
            scenario_dir=scenario_dir,
            agent_id=agent_id,
        )
    if stage == "nudge":
        return lambda message: evaluate_agent_nudge_response(
            message,
            scenario=scenario,
            agent_id=agent_id,
        )
    raise ValueError(f"Unsupported stage: {stage}")


def _make_mutator(stage: str) -> Callable[[str, random.Random], str]:
    if stage in {"local", "global"}:
        return _mutate_python_code
    if stage == "nudge":
        return _mutate_nudge_message
    raise ValueError(f"Unsupported stage: {stage}")


def _default_seed_directory(scenario_dir: Path, stage: str, agent_id: int) -> Path:
    return scenario_dir / stage / f"agent_{agent_id}" / "seeds"


def _store_best_candidate(stage: str, agent_dir: Path, candidate: Candidate) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    if stage in {"local", "global"}:
        output_path = agent_dir / "best_policy.py"
    else:
        output_path = agent_dir / "best_message.json"
    output_path.write_text(candidate.payload, encoding="utf-8")


def _write_history(agent_dir: Path, history: List[Dict[str, object]]) -> None:
    history_path = agent_dir / "evolution_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def run_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evolutionary optimisation for EV heuristics")
    parser.add_argument("stage", choices=["local", "global", "nudge"], help="Optimisation stage")
    parser.add_argument("--scenario-dir", default="data/energy_ev/scenario_1", help="Scenario directory")
    parser.add_argument("--agent", type=int, required=True, help="Agent identifier (1-indexed)")
    parser.add_argument("--seed-dir", help="Directory containing initial candidate files")
    parser.add_argument("--generations", type=int, default=4, help="Number of evolutionary generations")
    parser.add_argument("--population", type=int, default=6, help="Population size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args(argv)

    scenario_dir = Path(args.scenario_dir).resolve()
    scenario = load_scenario(scenario_dir / "scenario.json")
    agent_dir = scenario_dir / args.stage / f"agent_{args.agent}"

    seed_dir = Path(args.seed_dir) if args.seed_dir else _default_seed_directory(scenario_dir, args.stage, args.agent)
    initial_payloads = _load_payloads_from_directory(args.stage, seed_dir)

    evaluator = _make_evaluator(
        stage=args.stage,
        scenario=scenario,
        scenario_dir=scenario_dir,
        agent_id=args.agent,
    )
    mutator = _make_mutator(args.stage)

    strategy = EvolutionaryStrategy(
        evaluate=evaluator,
        mutate=mutator,
        population_size=args.population,
        seed=args.seed,
    )

    best, history = strategy.run(initial_payloads, args.generations)
    _store_best_candidate(args.stage, agent_dir, best)
    _write_history(agent_dir, history)

    print(json.dumps({"best_score": best.score, "detail": best.detail}, indent=2))


def main() -> None:
    run_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
