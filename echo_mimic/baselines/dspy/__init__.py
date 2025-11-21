"""DSPy baseline entry points for farm and energy domains."""

from typing import Optional

from echo_mimic.baselines.dspy.farm import main as _run_dspy_farm
from echo_mimic.baselines.dspy.global_baseline import main as _run_dspy_global
from echo_mimic.baselines.dspy.nudge import main as _run_dspy_nudge


def run_dspy_farm(*, model_name: Optional[str] = None, data_root: Optional[str] = None) -> None:
    """Run the DSPy heuristic baseline for local agents."""

    _run_dspy_farm(model_name=model_name, data_root=data_root)


def run_dspy_global(*, model_name: Optional[str] = None, data_root: Optional[str] = None) -> None:
    """Run the DSPy baseline for global coordination prompts."""

    _run_dspy_global(model_name=model_name, data_root=data_root)


def run_dspy_nudge(*, model_name: Optional[str] = None, data_root: Optional[str] = None) -> None:
    """Run the DSPy nudge baseline."""

    _run_dspy_nudge(model_name=model_name, data_root=data_root)


__all__ = ["run_dspy_farm", "run_dspy_global", "run_dspy_nudge"]
