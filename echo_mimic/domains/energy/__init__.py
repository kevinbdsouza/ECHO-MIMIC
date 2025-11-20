"""Energy domain orchestration wrappers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from energy_local_evo_strat import run as run_energy_local
from energy_global_evo_strat import run as run_energy_global
from energy_nudge_evo_strat import run as run_energy_nudge

from echo_mimic.baselines.autogen import AutoGenBaseline

logger = logging.getLogger(__name__)


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


class EnergyDomain:
    def __init__(self, config: EnergyDomainConfig):
        self.config = config
        self.autogen = AutoGenBaseline(domain="energy")

    def _echo_mimic_dispatch(self) -> None:
        if self.config.mode == "local":
            run_energy_local(
                population_size_value=self.config.population_size,
                num_generations_value=self.config.num_generations,
                init_value=self.config.init,
            )
        elif self.config.mode == "global":
            run_energy_global(
                population_size_value=self.config.population_size,
                num_generations_value=self.config.num_generations,
                init_value=self.config.init,
            )
        elif self.config.mode == "nudge":
            run_energy_nudge(
                population_size_value=self.config.population_size,
                num_generations_value=self.config.num_generations,
                inner_loop_size_value=self.config.inner_loop_size,
                init_value=self.config.init,
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

    def run(self) -> None:
        handlers: Dict[str, Callable[[], None]] = {
            "echo_mimic": self._echo_mimic_dispatch,
            "autogen": self._autogen_dispatch,
        }
        method = self.config.method.lower()
        if method not in handlers:
            raise ValueError(
                f"Unsupported energy method: {self.config.method}. Methods: {list(handlers)}"
            )
        logger.info("Running energy domain with method=%s mode=%s agent=%s model=%s", method, self.config.mode, self.config.agent_id, self.config.model)
        handlers[method]()
