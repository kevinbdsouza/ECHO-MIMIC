"""Utilities for working with evolutionary operators across strategies."""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


OPERATOR_NAMES: Sequence[str] = (
    "init",
    "mutate",
    "crossover",
    "evolve_1",
    "evolve_2",
    "reflect",
)


def make_operator_counts() -> dict[str, int]:
    return {op: 0 for op in OPERATOR_NAMES}


def make_operator_deltas() -> dict[str, float]:
    return {op: 0.0 for op in OPERATOR_NAMES}


def plot_best_trajectory_across_generations(
    best_history: Sequence[Mapping[str, object]],
    gen_dir: str,
    *,
    operator_sequence: Sequence[str] | None = None,
) -> None:
    base_op_list = tuple(operator_sequence or OPERATOR_NAMES)
    operator_map = {op: idx for idx, op in enumerate(base_op_list)}

    for entry in best_history:
        gen = entry["generation"]
        score = entry["score"]
        trajectory_list = entry["trajectory"]
        counts_dict = entry["counts"]
        fitness_deltas_dict = entry["fitness_deltas"]

        ops: list[str] = []
        deltas: list[float] = []

        for t in trajectory_list:
            idx = t.find("(")
            if idx == -1:
                op_name = t.strip()
                delta_str = "0.0"
            else:
                op_name = t[:idx].strip()
                delta_str = t[idx:].strip("()")

            try:
                delta_val = float(delta_str)
            except Exception:
                delta_val = 0.0

            ops.append(op_name)
            deltas.append(delta_val)

        x_vals = list(range(len(ops)))
        y_op = [operator_map.get(op, -1) for op in ops]

        plt.figure()
        plt.plot(x_vals, y_op, marker="o")
        plt.title(f"Best Candidate Trajectory (Gen {gen})\nScore={score:.4f}")
        plt.xlabel("Trajectory Step")
        plt.ylabel("Operators)")
        plt.yticks(list(range(len(base_op_list))), base_op_list)

        for i, (x, y) in enumerate(zip(x_vals, y_op)):
            d = deltas[i]
            plt.text(x, y, f"{d:+.3f}", fontsize=9, ha="left", va="bottom")

        outname = os.path.join(gen_dir, f"best_trajectory_gen_{gen}.png")
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()

        plt.figure()
        op_names = list(counts_dict.keys())
        usage_vals = [counts_dict[op] for op in op_names]
        plt.bar(op_names, usage_vals)
        plt.title(f"Operator Usage Counts - Gen {gen}, Score={score:.4f}")
        plt.xlabel("Operator")
        plt.ylabel("Count")
        outname = os.path.join(gen_dir, f"best_counts_gen_{gen}.png")
        plt.savefig(outname)
        plt.close()

        plt.figure()
        ops_list = list(fitness_deltas_dict.keys())
        deltas_list = [fitness_deltas_dict[op] for op in ops_list]
        plt.bar(ops_list, deltas_list)
        plt.axhline(y=0.0, color="gray", linestyle="--")
        plt.title(f"Operator Cumulative Deltas (Gen {gen})\nScore={score:.4f}")
        plt.xlabel("Operator")
        plt.ylabel("Cumulative Fitness Delta")
        plt.tight_layout()
        outname = os.path.join(gen_dir, f"best_fitness_deltas_gen_{gen}.png")
        plt.savefig(outname)
        plt.close()


def plot_population_operator_stats(
    population: Iterable[Mapping[str, object]],
    generation: int,
    gen_dir: str,
    *,
    operator_sequence: Sequence[str] | None = None,
) -> None:
    operator_list = tuple(operator_sequence or OPERATOR_NAMES)
    usage_sums = {op: 0 for op in operator_list}
    delta_sums = {op: 0.0 for op in operator_list}

    for cand in population:
        counts = cand.get("counts", {})
        deltas = cand.get("fitness_deltas", {})
        for op in operator_list:
            usage_sums[op] += counts.get(op, 0)
            delta_sums[op] += deltas.get(op, 0.0)

    plt.figure()
    ops_list = list(usage_sums.keys())
    usage_vals = [usage_sums[op] for op in ops_list]
    plt.bar(ops_list, usage_vals)
    plt.title(f"All-Pop Operator Usage (Gen {generation})")
    plt.xlabel("Operator")
    plt.ylabel("Usage Count")
    outname = os.path.join(gen_dir, f"operator_usage_all_gen_{generation}.png")
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

    plt.figure()
    delta_vals = [delta_sums[op] for op in ops_list]
    plt.bar(ops_list, delta_vals)
    plt.axhline(y=0.0, color="gray", linestyle="--")
    plt.title(f"All-Pop Operator Deltas (Gen {generation})")
    plt.xlabel("Operator")
    plt.ylabel("Sum of Fitness Deltas")
    outname = os.path.join(gen_dir, f"operator_deltas_all_gen_{generation}.png")
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()


__all__ = [
    "OPERATOR_NAMES",
    "make_operator_counts",
    "make_operator_deltas",
    "plot_best_trajectory_across_generations",
    "plot_population_operator_stats",
]
