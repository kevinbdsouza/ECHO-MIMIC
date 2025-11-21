"""Shared helpers for computing Radon-based code metrics."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Any

import pandas as pd
from radon.raw import analyze
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit


HALSTEAD_FIELDS = (
    ("h1", "halstead_h1"),
    ("h2", "halstead_h2"),
    ("N1", "halstead_N1"),
    ("N2", "halstead_N2"),
    ("vocabulary", "halstead_vocabulary"),
    ("length", "halstead_length"),
    ("volume", "halstead_volume"),
    ("difficulty", "halstead_difficulty"),
    ("effort", "halstead_effort"),
    ("time", "halstead_time"),
    ("bugs", "halstead_bugs"),
)


def _empty_halstead_totals() -> dict[str, float]:
    return {dest: 0.0 for _, dest in HALSTEAD_FIELDS}


def _aggregate_halstead(halstead_results: Any) -> dict[str, float]:
    """Extract aggregate Halstead metrics from Radon's result object."""

    totals = _empty_halstead_totals()
    if not halstead_results:
        return totals

    total = getattr(halstead_results, "total", None)
    if total is not None:
        for src, dest in HALSTEAD_FIELDS:
            totals[dest] = float(getattr(total, src, 0.0) or 0.0)
        return totals

    try:
        iterator = iter(halstead_results)
    except TypeError:
        return totals

    for item in iterator:
        for src, dest in HALSTEAD_FIELDS:
            totals[dest] += float(getattr(item, src, 0.0) or 0.0)

    return totals


def _compute_metrics_row(code: str, score: float) -> dict[str, float] | None:
    if not code or not isinstance(code, str):
        return None

    try:
        raw_metrics = analyze(code)
    except Exception:
        return None

    try:
        complexities = cc_visit(code)
    except Exception:
        complexities = []

    if complexities:
        avg_cyclomatic_complexity = sum(c.complexity for c in complexities) / len(complexities)
    else:
        avg_cyclomatic_complexity = 0.0

    try:
        halstead_results = h_visit(code)
    except Exception:
        halstead_results = None

    halstead_totals = _aggregate_halstead(halstead_results)

    try:
        mi_value = mi_visit(code, multi=True)
    except Exception:
        mi_value = 0.0

    row = {
        "fitness_score": score,
        "loc": raw_metrics.loc,
        "lloc": raw_metrics.lloc,
        "sloc": raw_metrics.sloc,
        "comment": raw_metrics.comments,
        "multi": raw_metrics.multi,
        "blank": raw_metrics.blank,
        "avg_cyclomatic_complexity": avg_cyclomatic_complexity,
        "maintainability_index": mi_value,
    }
    row.update(halstead_totals)
    return row


def compute_radon_metrics(
    old_df: pd.DataFrame,
    codes: Iterable[str],
    scores: Iterable[float],
) -> pd.DataFrame:
    """Append Radon metrics for the provided code blocks to ``old_df``."""

    new_rows = []
    for code, score in zip(codes, scores):
        row = _compute_metrics_row(code, score)
        if row is not None:
            new_rows.append(row)

    if not new_rows:
        return old_df.copy()

    new_df = pd.DataFrame(new_rows)
    if old_df.empty:
        return new_df.reset_index(drop=True)

    return pd.concat([old_df, new_df], ignore_index=True)


def compute_radon_metrics_from_population(
    old_df: pd.DataFrame,
    population: Iterable[Mapping[str, Any]],
    *,
    code_key: str = "code",
    score_key: str = "score",
) -> pd.DataFrame:
    """Helper for populations that store ``code`` and ``score`` together."""

    codes: list[str] = []
    scores: list[float] = []

    for candidate in population:
        code = candidate.get(code_key)
        if not code or not isinstance(code, str):
            continue
        codes.append(code)
        scores.append(candidate.get(score_key, 0.0))

    return compute_radon_metrics(old_df, codes, scores)


__all__ = [
    "compute_radon_metrics",
    "compute_radon_metrics_from_population",
]
