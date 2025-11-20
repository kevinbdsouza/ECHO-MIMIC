"""DSPy baseline entry points for farm domain."""
from echo_mimic.baselines.dspy.farm import main as run_dspy_farm
from echo_mimic.baselines.dspy.global_baseline import main as run_dspy_global
from echo_mimic.baselines.dspy.nudge import main as run_dspy_nudge

__all__ = ["run_dspy_farm", "run_dspy_global", "run_dspy_nudge"]
