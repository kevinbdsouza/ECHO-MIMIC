"""Gemini-powered evolutionary search for personalised EV nudges."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import google.generativeai as genai

from echo_mimic.common import build_model, configure_genai, ensure_rate_limiter, is_openai_model
from echo_mimic.common.operators import (
    make_operator_counts,
    make_operator_deltas,
    plot_best_trajectory_across_generations,
    plot_population_operator_stats,
)
from echo_mimic.config import Config
from .prompts import build_stage_four_prompt
from .evaluation import evaluate_agent_nudge_response
from .scenario import load_scenario
from echo_mimic.common.rate_limiter import send_message_with_retry

cfg = Config()
rate_limiter = ensure_rate_limiter(cfg)

MESSAGE_GUIDELINES = (
    "- Respond with valid JSON containing persona, recommended_usage, and message keys.\n"
    "- The persona must match the agent profile in the prompt.\n"
    "- recommended_usage must be a 7-element list of usage vectors, each containing four floats in [0, 1].\n"
    "- Keep the message concise, factual, and aligned with feeder and comfort constraints.\n"
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def init_model(model_name: str):
    if not is_openai_model(model_name):
        configure_genai()
    system_instructions = (
        "You craft persuasive yet respectful energy-behaviour nudges. "
        "Given persona context and policy code, output JSON with persona, recommended_usage (seven usage vectors with slot values between 0 and 1), and message."
    )
    return build_model(model_name, system_instructions, ensure_configured=False)


def _completion_text(completion: object) -> str:
    if hasattr(completion, "text") and completion.text:
        return completion.text
    parts: List[str] = []
    for part in getattr(completion, "parts", []):
        text = getattr(part, "text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _normalise_message(raw_text: str) -> str:
    """Coerce model output into compact JSON, tolerating stray formatting."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Model returned an empty response for the nudge request")

    # Try the raw text first, then fall back to common fence/extra-text patterns.
    candidates = [text]
    if text.startswith("```"):
        unwrapped = text
        for prefix in ("```json", "```"):
            if unwrapped.startswith(prefix):
                unwrapped = unwrapped[len(prefix):]
                break
        if unwrapped.endswith("```"):
            unwrapped = unwrapped[:-3]
        candidates.append(unwrapped.strip())

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1].strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Model response is not valid JSON: {text[:200]}")


# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def _make_candidate(message: str) -> Dict[str, object]:
    counts = make_operator_counts()
    counts["init"] += 1
    return {
        "message": message,
        "score": 0.0,
        "detail": {},
        "counts": counts,
        "fitness_deltas": make_operator_deltas(),
        "trajectory": ["init(0.0000)"],
    }


def _load_population(pop_dir: Path, limit: int) -> List[Dict[str, object]]:
    population: List[Dict[str, object]] = []
    for path in sorted(pop_dir.glob("*.json"))[:limit]:
        message = path.read_text(encoding="utf-8").strip()
        if message:
            population.append(_make_candidate(message))
    return population


def _write_population(pop_dir: Path, candidates: Sequence[Dict[str, object]]) -> None:
    pop_dir.mkdir(parents=True, exist_ok=True)
    for idx, cand in enumerate(candidates, start=1):
        path = pop_dir / f"candidate_{idx:02d}.json"
        path.write_text(cand["message"] + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt operations
# ---------------------------------------------------------------------------

def _request_message(model: genai.GenerativeModel, prompt: str) -> str:
    session = model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, rate_limiter)
    text = _completion_text(completion).strip()
    if not text:
        raise RuntimeError("Model produced empty response")
    return _normalise_message(text)


def _mutate_message(prompt: str, parent: Dict[str, object], model: genai.GenerativeModel) -> Dict[str, object]:
    op_prompt = (
        prompt
        + "\n\n"
        + MESSAGE_GUIDELINES
        + "\nRefine the existing JSON nudge by strengthening its reasoning while keeping the structure identical."
        + " Return only the updated JSON object.\n\nCurrent message:\n"
        + json.dumps(json.loads(parent["message"]), indent=2, sort_keys=True)
    )
    child_message = _request_message(model, op_prompt)
    return _copy_child(parent, child_message, "mutate")


def _crossover_message(prompt: str, parent_a: Dict[str, object], parent_b: Dict[str, object], model: genai.GenerativeModel) -> Dict[str, object]:
    op_prompt = (
        prompt
        + "\n\n"
        + MESSAGE_GUIDELINES
        + "\nBlend the strongest persuasive elements from both JSON nudges into a single improved message."
        + " Return only the merged JSON.\n\nParent A:\n"
        + json.dumps(json.loads(parent_a["message"]), indent=2, sort_keys=True)
        + "\n\nParent B:\n"
        + json.dumps(json.loads(parent_b["message"]), indent=2, sort_keys=True)
    )
    child_message = _request_message(model, op_prompt)
    dominant = parent_a if parent_a["score"] >= parent_b["score"] else parent_b
    return _copy_child(dominant, child_message, "crossover")


def _evolve_message(
    prompt: str,
    parent_a: Dict[str, object],
    parent_b: Dict[str, object],
    model: genai.GenerativeModel,
    *,
    explore_new: bool,
) -> Dict[str, object]:
    if explore_new:
        instruction = "Design a substantially different JSON nudge exploring a new persuasive framing."
        op_name = "evolve_1"
    else:
        instruction = (
            "Identify shared themes between both nudges and extend them with deeper reasoning without copying."
        )
        op_name = "evolve_2"
    op_prompt = (
        prompt
        + "\n\n"
        + MESSAGE_GUIDELINES
        + "\n"
        + instruction
        + " Return only the resulting JSON.\n\nParent A:\n"
        + json.dumps(json.loads(parent_a["message"]), indent=2, sort_keys=True)
        + "\n\nParent B:\n"
        + json.dumps(json.loads(parent_b["message"]), indent=2, sort_keys=True)
    )
    child_message = _request_message(model, op_prompt)
    dominant = parent_a if parent_a["score"] >= parent_b["score"] else parent_b
    return _copy_child(dominant, child_message, op_name)


def _reflect_messages(prompt: str, population: Sequence[Dict[str, object]], model: genai.GenerativeModel) -> List[Dict[str, object]]:
    top_candidates = sorted(population, key=lambda cand: cand["score"], reverse=True)[:5]
    if not top_candidates:
        return []
    summary_lines = []
    for idx, cand in enumerate(top_candidates, start=1):
        summary_lines.append(
            f"Message {idx} (score={cand['score']:.4f}):\n{json.dumps(json.loads(cand['message']), indent=2, sort_keys=True)}"
        )
    op_prompt = (
        prompt
        + "\n\n"
        + MESSAGE_GUIDELINES
        + "\nReflect on the best nudges and craft a higher-impact JSON message."
        + " Explain reasoning internally but return only the JSON.\n\n"
        + "\n\n".join(summary_lines)
    )
    child_message = _request_message(model, op_prompt)
    dominant = top_candidates[0]
    return [_copy_child(dominant, child_message, "reflect")]


def _copy_child(parent: Dict[str, object], message: str, op: str) -> Dict[str, object]:
    return {
        "message": message,
        "score": 0.0,
        "detail": {},
        "counts": parent["counts"].copy(),
        "fitness_deltas": parent["fitness_deltas"].copy(),
        "trajectory": parent["trajectory"][:],
        "pending_op": op,
        "pending_parent_score": parent["score"],
    }


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

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

    agent_dir = scenario_dir / "nudge" / f"agent_{agent_id}"
    pop_dir = agent_dir / "population"
    gen_dir = agent_dir / "generations"
    agent_dir.mkdir(parents=True, exist_ok=True)
    pop_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_stage_four_prompt(scenario, agent_id, scenario_dir=scenario_dir)
    agent_dir.joinpath("prompt_input.txt").write_text(prompt, encoding="utf-8")

    model = init_model(model_name or cfg.lm)
    rng = random.Random(seed)

    if init:
        seeds: List[Dict[str, object]] = []
        for _ in range(population_size):
            message = _request_message(
                model,
                prompt
                + "\n\n"
                + MESSAGE_GUIDELINES
                + "\nReturn only one JSON object that satisfies the constraints.",
            )
            seeds.append(_make_candidate(message))
        _write_population(pop_dir, seeds)

    population = _load_population(pop_dir, population_size)
    if not population:
        raise RuntimeError("No initial population found. Run with --init to generate seeds.")

    def evaluate(candidate: Dict[str, object]) -> Dict[str, object]:
        score, detail = evaluate_agent_nudge_response(
            candidate["message"],
            scenario=scenario,
            agent_id=agent_id,
        )
        candidate["score"] = score
        candidate["detail"] = detail
        op = candidate.pop("pending_op", None)
        parent_score = candidate.pop("pending_parent_score", None)
        if op is not None and parent_score is not None:
            delta = score - parent_score
            candidate["counts"][op] += 1
            candidate["fitness_deltas"][op] += delta
            candidate["trajectory"].append(f"{op}({delta:+.4f})")
        return candidate

    population = [evaluate(candidate) for candidate in population]
    history: List[Dict[str, object]] = []

    for generation in range(generations + 1):
        best = max(population, key=lambda cand: cand["score"])
        history.append(
            {
                "generation": generation,
                # Keep both 'score' and 'best_score' for compatibility with plotting helpers.
                "score": best["score"],
                "best_score": best["score"],
                "best_detail": best.get("detail"),
                "trajectory": best.get("trajectory"),
                "counts": best.get("counts"),
                "fitness_deltas": best.get("fitness_deltas"),
            }
        )
        plot_best_trajectory_across_generations(history[-1:], str(gen_dir))
        plot_population_operator_stats(population, generation, str(gen_dir))
        _write_population(pop_dir, population[:population_size])

        if generation == generations:
            break

        offspring: List[Dict[str, object]] = []

        # Always allow selection to consider the unchanged population.
        for parent in population:
            offspring.append(
                {
                    "message": parent["message"],
                    "score": 0.0,
                    "detail": {},
                    "counts": parent["counts"].copy(),
                    "fitness_deltas": parent["fitness_deltas"].copy(),
                    "trajectory": parent["trajectory"][:],
                }
            )

        # For each inner loop slot, attempt every operator variant.
        for _ in range(inner_loop_size):
            parent_a = rng.choice(population)
            parent_b = rng.choice(population)
            for op_name, builder in (
                ("mutate", lambda: _mutate_message(prompt, parent_a, model)),
                ("crossover", lambda: _crossover_message(prompt, parent_a, parent_b, model)),
                ("evolve_1", lambda: _evolve_message(prompt, parent_a, parent_b, model, explore_new=True)),
                ("evolve_2", lambda: _evolve_message(prompt, parent_a, parent_b, model, explore_new=False)),
            ):
                try:
                    offspring.append(builder())
                except Exception as exc:
                    print(f"[warn] {op_name} failed: {exc}")

        offspring = [evaluate(child) for child in offspring]
        reflect_children = _reflect_messages(prompt, population, model)
        reflect_children = [evaluate(child) for child in reflect_children]

        combined = offspring + reflect_children
        combined.sort(key=lambda cand: cand["score"], reverse=True)
        population = combined[: population_size]

    best = max(population, key=lambda cand: cand["score"])

    agent_dir.joinpath("best_message.json").write_text(best["message"] + "\n", encoding="utf-8")
    agent_dir.joinpath("best_message_detail.json").write_text(
        json.dumps(best.get("detail", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    agent_dir.joinpath("evo_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Completed nudge evolution for agent {agent_id}: best score {best['score']:.3f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Gemini ES for EV nudges.")
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("data/energy_ev/scenario_1"),
        help="Directory containing scenario.json and agent subfolders.",
    )
    parser.add_argument("--agent-id", type=int, default=5, help="Target agent id to optimise.")
    parser.add_argument("--population-size", type=int, default=5, help="Population size for the ES loop.")
    parser.add_argument("--generations", type=int, default=1, help="Number of generations to evolve.")
    parser.add_argument(
        "--inner-loop-size",
        type=int,
        default=1,
        help="Number of offspring attempts per generation.",
    )
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
        inner_loop_size=max(1, args.inner_loop_size),
        seed=args.seed,
        init=not args.no_init,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
