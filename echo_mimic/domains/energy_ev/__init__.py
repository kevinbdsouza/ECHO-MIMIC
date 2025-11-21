"""Utilities and orchestration helpers for the carbon-aware EV charging domain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .energy_local_evo_strat import run as run_energy_local
from .energy_global_evo_strat import run as run_energy_global
from .energy_nudge_evo_strat import run as run_energy_nudge

from echo_mimic.baselines.autogen import AutoGenBaseline
from echo_mimic.baselines.dspy import (
    run_dspy_farm as run_dspy_local_baseline,
    run_dspy_global as run_dspy_global_baseline,
    run_dspy_nudge as run_dspy_nudge_baseline,
)

from .scenario import (
    AgentConfig,
    EVScenario,
    load_scenario,
    compute_local_cost,
    compute_global_cost,
    enumerate_global_optimum,
    dump_ground_truth,
)

from .prompts import (
    build_stage_one_prompt,
    build_stage_two_prompt,
    build_stage_two_prompts,
    build_stage_three_prompt,
    build_stage_three_prompts,
    build_stage_four_prompt,
    build_stage_four_prompts,
)

from .evaluation import (
    evaluate_local_agent_policy_script,
    evaluate_global_agent_policy_script,
    evaluate_agent_nudge_response,
)

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_DIR = Path("data/energy_ev/scenario_1")
DEFAULT_SEEDS = {"local": 13, "global": 17, "nudge": 23}


@dataclass
class EnergyDomainConfig:
    mode: str
    method: str
    agent_id: Optional[str]
    model: Optional[str]
    population_size: int
    num_generations: int
    inner_loop_size: int
    init: bool = True
    scenario_dir: Path = DEFAULT_SCENARIO_DIR
    seed: Optional[int] = None


class EnergyDomain:
    """Orchestrates runs for the energy EV scenarios."""

    def __init__(self, config: EnergyDomainConfig):
        self.config = config
        self.autogen = AutoGenBaseline(domain="energy_ev")

    def _resolve_agent_id(self) -> int:
        if self.config.agent_id is None:
            return 1
        try:
            return int(self.config.agent_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Energy domain requires an integer agent id, received {self.config.agent_id!r}"
            ) from exc

    def _resolve_seed(self) -> int:
        if self.config.seed is not None:
            return self.config.seed
        return DEFAULT_SEEDS.get(self.config.mode.lower(), DEFAULT_SEEDS["local"])

    def _resolve_scenario_dir(self) -> Path:
        return Path(self.config.scenario_dir).expanduser()

    def _resolve_dspy_dataset_root(self) -> Path:
        """Return the dataset root DSPy baselines should inspect."""

        scenario_dir = self._resolve_scenario_dir()
        # When targeting a specific scenario_* folder, use its parent so DSPy can
        # iterate over all agents/scenarios under data/energy_ev.
        if scenario_dir.name.startswith("scenario_"):
            parent = scenario_dir.parent
            if parent.exists():
                return parent
        return scenario_dir

    def _echo_mimic_dispatch(self) -> None:
        scenario_dir = self._resolve_scenario_dir()
        agent_id = self._resolve_agent_id()
        seed = self._resolve_seed()
        mode = self.config.mode.lower()
        if mode == "local":
            run_energy_local(
                scenario_dir=scenario_dir,
                agent_id=agent_id,
                population_size=self.config.population_size,
                generations=self.config.num_generations,
                inner_loop_size=self.config.inner_loop_size,
                seed=seed,
                init=self.config.init,
                model_name=self.config.model,
            )
        elif mode == "global":
            run_energy_global(
                scenario_dir=scenario_dir,
                agent_id=agent_id,
                population_size=self.config.population_size,
                generations=self.config.num_generations,
                inner_loop_size=self.config.inner_loop_size,
                seed=seed,
                init=self.config.init,
                model_name=self.config.model,
            )
        elif mode == "nudge":
            run_energy_nudge(
                scenario_dir=scenario_dir,
                agent_id=agent_id,
                population_size=self.config.population_size,
                generations=self.config.num_generations,
                inner_loop_size=self.config.inner_loop_size,
                seed=seed,
                init=self.config.init,
                model_name=self.config.model,
            )
        else:
            raise ValueError(f"Unsupported energy mode: {self.config.mode}")

    def _autogen_dispatch(self) -> None:
        self.autogen.run(
            mode=self.config.mode,
            agent_id=self.config.agent_id,
            model=self.config.model,
            data_hint="data/energy_ev/",
        )

    def _dspy_dispatch(self) -> None:
        dataset_root = str(self._resolve_dspy_dataset_root())
        model_name = self.config.model
        mode = self.config.mode.lower()
        if mode == "local":
            run_dspy_local_baseline(model_name=model_name, data_root=dataset_root)
        elif mode == "global":
            run_dspy_global_baseline(model_name=model_name, data_root=dataset_root)
        elif mode == "nudge":
            run_dspy_nudge_baseline(model_name=model_name, data_root=dataset_root)
        else:
            raise ValueError(f"Unsupported DSPy energy mode: {self.config.mode}")

    def run(self) -> None:
        handlers: Dict[str, Callable[[], None]] = {
            "echo_mimic": self._echo_mimic_dispatch,
            "dspy": self._dspy_dispatch,
            "autogen": self._autogen_dispatch,
        }
        method = self.config.method.lower()
        if method not in handlers:
            raise ValueError(
                f"Unsupported energy method: {self.config.method}. Methods: {list(handlers)}"
            )
        logger.info(
            "Running energy domain with method=%s mode=%s agent=%s model=%s",
            method,
            self.config.mode,
            self.config.agent_id,
            self.config.model,
        )
        handlers[method]()


__all__ = [
    "AgentConfig",
    "EVScenario",
    "load_scenario",
    "compute_local_cost",
    "compute_global_cost",
    "enumerate_global_optimum",
    "dump_ground_truth",
    "build_stage_one_prompt",
    "build_stage_two_prompt",
    "build_stage_two_prompts",
    "build_stage_three_prompt",
    "build_stage_three_prompts",
    "build_stage_four_prompt",
    "build_stage_four_prompts",
    "evaluate_local_agent_policy_script",
    "evaluate_global_agent_policy_script",
    "evaluate_agent_nudge_response",
    "EnergyDomainConfig",
    "EnergyDomain",
]
