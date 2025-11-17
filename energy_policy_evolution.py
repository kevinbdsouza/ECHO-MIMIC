from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import google.generativeai as genai

from echo_mimic.common import (
    CommandOutputCapture,
    extract_python_code,
    make_code_validator,
)
from echo_mimic.common.operators import (
    make_operator_counts,
    make_operator_deltas,
    plot_best_trajectory_across_generations,
    plot_population_operator_stats,
)
from echo_mimic.rate_limiter import send_message_with_retry


def _completion_text(completion: genai.types.GenerateContentResponse) -> str:
    if hasattr(completion, "text") and completion.text:
        return completion.text
    parts: List[str] = []
    for part in getattr(completion, "parts", []):
        text = getattr(part, "text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)


@dataclass
class EnergyPolicyContext:
    """Immutable configuration used by the runner."""

    stage_name: str
    prompt: str
    guidelines: str
    agent_dir: Path
    evaluator: Callable[[str], tuple[float, Dict[str, object]]]
    heur_model: genai.GenerativeModel
    fix_model: genai.GenerativeModel
    rate_limiter
    script_name: str = "policy.py"


class EnergyPolicyEvolutionRunner:
    """Multi-operator evolutionary loop tailored to policy.py search."""

    def __init__(self, context: EnergyPolicyContext, *, seed: int = 0) -> None:
        self._context = context
        self._population_dir = context.agent_dir / "population"
        self._generation_dir = context.agent_dir / "generations"
        self._population_dir.mkdir(parents=True, exist_ok=True)
        self._generation_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)

        capture = CommandOutputCapture()
        self._validator = make_code_validator(
            workdir=context.agent_dir,
            capture=capture,
            fix_model=context.fix_model,
            rate_limiter=context.rate_limiter,
            default_script=context.script_name,
            default_attempts=2,
        )

    # ------------------------------------------------------------------
    # Seed management
    # ------------------------------------------------------------------
    def generate_initial_population(self, population_size: int, *, overwrite: bool = False) -> None:
        if overwrite:
            for path in self._population_dir.glob("*"):
                path.unlink()
        for idx in range(1, population_size + 1):
            code = self._request_policy_code(
                self._context.prompt
                + "\n\nRespond with a complete Python implementation of policy.py."
                + " Do not include commentary or markdown fences."
            )
            candidate_path = self._population_dir / f"candidate_{idx:02d}.py"
            candidate_path.write_text(code, encoding="utf-8")

    def load_population(self, population_size: int) -> List[Dict[str, object]]:
        population: List[Dict[str, object]] = []
        files = sorted(self._population_dir.glob("*.py"))[:population_size]
        for path in files:
            code = path.read_text(encoding="utf-8")
            population.append(self._make_candidate(code))
        return population

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        population_size: int,
        generations: int,
        inner_loop_size: int,
    ) -> Dict[str, object]:
        population = self.load_population(population_size)
        if not population:
            raise RuntimeError("No initial population available. Run generate_initial_population first.")

        population = self._evaluate_population(population)
        history: List[Dict[str, object]] = [self._snapshot(population, generation=0)]

        for generation in range(1, generations + 1):
            offspring = self._produce_offspring(population, inner_loop_size)
            offspring = self._evaluate_population(offspring)

            reflect_children = self._run_reflect(population)
            if reflect_children:
                reflect_children = self._evaluate_population(reflect_children)
                offspring.extend(reflect_children)

            combined = self._select_population(population + offspring, population_size)
            population = combined
            history.append(self._snapshot(population, generation))

            self._write_population(population)
            self._write_generation_summary(generation, population)

        best_candidate = max(population, key=lambda cand: cand["score"])
        return {"best": best_candidate, "history": history}

    # ------------------------------------------------------------------
    # Operator helpers
    # ------------------------------------------------------------------
    def _produce_offspring(
        self,
        population: Sequence[Dict[str, object]],
        inner_loop_size: int,
    ) -> List[Dict[str, object]]:
        children: List[Dict[str, object]] = []
        if not population:
            return children
        operator_cycle = ("mutate", "crossover", "evolve_1", "evolve_2")
        while len(children) < inner_loop_size:
            op_name = operator_cycle[len(children) % len(operator_cycle)]
            try:
                if op_name == "mutate":
                    parent = self._rng.choice(population)
                    child = self._apply_mutate(parent)
                else:
                    parent1 = self._rng.choice(population)
                    parent2 = self._rng.choice(population)
                    if op_name == "crossover":
                        child = self._apply_crossover(parent1, parent2)
                    elif op_name == "evolve_1":
                        child = self._apply_evolve(parent1, parent2, explore_new=True)
                    else:
                        child = self._apply_evolve(parent1, parent2, explore_new=False)
                children.append(child)
            except Exception as exc:  # pragma: no cover - defensive
                # Skip failed operator invocations but keep the loop running
                print(f"[warn] operator {op_name} failed: {exc}")
                continue
        return children

    def _apply_mutate(self, parent: Dict[str, object]) -> Dict[str, object]:
        prompt = self._base_prompt(
            "You are revising the existing policy implementation."
            " Introduce precise, well-motivated improvements while"
            " preserving determinism and required file IO."
            " Return only the updated policy.py contents."
        )
        prompt += "\n\nCurrent policy.py:\n```python\n" + parent["code"] + "\n```"
        code = self._request_policy_code(prompt)
        return self._copy_child(parent, code, "mutate")

    def _apply_crossover(self, parent1: Dict[str, object], parent2: Dict[str, object]) -> Dict[str, object]:
        prompt = self._base_prompt(
            "You are fusing two complementary heuristic implementations."
            " Merge their strongest reasoning patterns into a single coherent policy."
            " Avoid duplicating logic and keep the output deterministic."
        )
        prompt += (
            "\n\nParent A policy.py:\n```python\n"
            + parent1["code"]
            + "\n```\n\nParent B policy.py:\n```python\n"
            + parent2["code"]
            + "\n```"
        )
        prompt += "\n\nReturn the integrated policy.py only."
        code = self._request_policy_code(prompt)
        dominant = self._dominant_parent(parent1, parent2)
        return self._copy_child(dominant, code, "crossover")

    def _apply_evolve(
        self,
        parent1: Dict[str, object],
        parent2: Dict[str, object],
        *,
        explore_new: bool,
    ) -> Dict[str, object]:
        if explore_new:
            instruction = (
                "Derive a novel policy that intentionally departs from both parents"
                " while still satisfying all requirements."
            )
            op_name = "evolve_1"
        else:
            instruction = (
                "Identify the shared planning principles between both parents and"
                " extend them with fresh reasoning that improves robustness."
            )
            op_name = "evolve_2"
        prompt = self._base_prompt(instruction)
        prompt += (
            "\n\nParent A policy.py:\n```python\n"
            + parent1["code"]
            + "\n```\n\nParent B policy.py:\n```python\n"
            + parent2["code"]
            + "\n```"
        )
        prompt += "\n\nReturn the new complete policy.py only."
        code = self._request_policy_code(prompt)
        dominant = self._dominant_parent(parent1, parent2)
        return self._copy_child(dominant, code, op_name)

    def _run_reflect(self, population: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        top_candidates = sorted(population, key=lambda cand: cand["score"], reverse=True)[:5]
        if not top_candidates:
            return []
        summaries: List[str] = []
        for idx, cand in enumerate(top_candidates, start=1):
            summaries.append(
                f"Candidate {idx} (score={cand['score']:.4f}):\n{cand['code']}"
            )
        prompt = self._base_prompt(
            "Reflect on the best candidates and craft a new, higher-scoring policy."
            " Use the observations to motivate improvements without copying verbatim."
        )
        prompt += "\n\nTop candidate summaries:\n" + "\n\n".join(summaries)
        prompt += "\n\nReturn only the new policy.py."
        code = self._request_policy_code(prompt)
        dominant = top_candidates[0]
        child = self._copy_child(dominant, code, "reflect")
        return [child]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate_population(self, population: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        evaluated: List[Dict[str, object]] = []
        for candidate in population:
            score, detail = self._context.evaluator(candidate["code"])
            candidate["score"] = score
            candidate["detail"] = detail
            op = candidate.pop("pending_op", None)
            parent_score = candidate.pop("pending_parent_score", None)
            if op is not None and parent_score is not None:
                delta = score - parent_score
                candidate.setdefault("counts", make_operator_counts())
                candidate.setdefault("fitness_deltas", make_operator_deltas())
                candidate.setdefault("trajectory", ["init(0.0000)"])
                candidate["counts"][op] += 1
                candidate["fitness_deltas"][op] += delta
                candidate["trajectory"].append(f"{op}({delta:+.4f})")
            evaluated.append(candidate)
        return evaluated

    def _select_population(
        self,
        population: Sequence[Dict[str, object]],
        population_size: int,
    ) -> List[Dict[str, object]]:
        ordered = sorted(population, key=lambda cand: cand["score"], reverse=True)
        return ordered[:population_size]

    # ------------------------------------------------------------------
    # Prompting helpers
    # ------------------------------------------------------------------
    def _base_prompt(self, operator_instruction: str) -> str:
        return (
            self._context.prompt
            + "\n\nImplementation requirements:\n"
            + self._context.guidelines
            + "\n\n"
            + operator_instruction
        )

    def _request_policy_code(self, prompt: str) -> str:
        session = self._context.heur_model.start_chat(history=[])
        completion = send_message_with_retry(session, prompt, self._context.rate_limiter)
        response_text = _completion_text(completion)
        code = extract_python_code(response_text) or response_text
        code = code.strip()
        if not code:
            raise RuntimeError("LLM produced empty policy code")
        return self._validator(code, script_name=self._context.script_name, max_attempts=2)

    def _make_candidate(self, code: str) -> Dict[str, object]:
        counts = make_operator_counts()
        counts["init"] += 1
        return {
            "code": code,
            "score": 0.0,
            "counts": counts,
            "fitness_deltas": make_operator_deltas(),
            "trajectory": ["init(0.0000)"],
        }

    def _copy_child(self, parent: Dict[str, object], code: str, op: str) -> Dict[str, object]:
        child = {
            "code": code,
            "score": 0.0,
            "counts": parent["counts"].copy(),
            "fitness_deltas": parent["fitness_deltas"].copy(),
            "trajectory": parent["trajectory"][:],
            "pending_op": op,
            "pending_parent_score": parent["score"],
        }
        return child

    def _dominant_parent(
        self,
        parent1: Dict[str, object],
        parent2: Dict[str, object],
    ) -> Dict[str, object]:
        if parent1["score"] >= parent2["score"]:
            return parent1
        return parent2

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _write_population(self, population: Sequence[Dict[str, object]]) -> None:
        for idx, candidate in enumerate(population, start=1):
            path = self._population_dir / f"candidate_{idx:02d}.py"
            path.write_text(candidate["code"], encoding="utf-8")

    def _write_generation_summary(
        self,
        generation: int,
        population: Sequence[Dict[str, object]],
    ) -> None:
        summary_path = self._generation_dir / f"population_gen_{generation:02d}.json"
        payload = [
            {
                "score": cand["score"],
                "trajectory": cand.get("trajectory", []),
                "counts": cand.get("counts", {}),
                "fitness_deltas": cand.get("fitness_deltas", {}),
            }
            for cand in population
        ]
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        plot_population_operator_stats(population, generation, str(self._generation_dir))

    def _snapshot(self, population: Sequence[Dict[str, object]], generation: int) -> Dict[str, object]:
        scores = [cand["score"] for cand in population]
        best = max(population, key=lambda cand: cand["score"]) if population else None
        entry = {
            "generation": generation,
            "population_size": len(population),
            "best_score": best["score"] if best else None,
            "mean_score": statistics.fmean(scores) if scores else None,
            "best_detail": best.get("detail") if best else None,
            "trajectory": best.get("trajectory") if best else None,
            "counts": best.get("counts") if best else None,
            "fitness_deltas": best.get("fitness_deltas") if best else None,
        }
        if best:
            plot_best_trajectory_across_generations(
                [entry],
                str(self._generation_dir),
            )
        return entry


__all__ = [
    "EnergyPolicyContext",
    "EnergyPolicyEvolutionRunner",
]
