"""Unified orchestration for running farm and energy domains."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from echo_mimic.domains.energy_ev import EnergyDomain, EnergyDomainConfig
from echo_mimic.domains.farm import FarmDomain, FarmDomainConfig

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    domain: str
    mode: str
    method: str
    agent_id: Optional[str]
    model: Optional[str]
    population_size: int = 25
    num_generations: int = 25
    inner_loop_size: int = 10
    use_template: bool = False
    use_hint: bool = True
    halstead_metrics: bool = False
    init: bool = True


class Orchestrator:
    def __init__(self, config: RunConfig):
        self.config = config

    def run(self) -> None:
        domain = self.config.domain.lower()
        if domain == "farm":
            farm = FarmDomain(
                FarmDomainConfig(
                    mode=self.config.mode,
                    method=self.config.method,
                    agent_id=self.config.agent_id,
                    model=self.config.model,
                    population_size=self.config.population_size,
                    num_generations=self.config.num_generations,
                    inner_loop_size=self.config.inner_loop_size,
                    use_template=self.config.use_template,
                    use_hint=self.config.use_hint,
                    halstead_metrics=self.config.halstead_metrics,
                    init=self.config.init,
                )
            )
            farm.run()
        elif domain in {"energy", "energy_ev"}:
            energy = EnergyDomain(
                EnergyDomainConfig(
                    mode=self.config.mode,
                    method=self.config.method,
                    agent_id=self.config.agent_id,
                    model=self.config.model,
                    population_size=self.config.population_size,
                    num_generations=self.config.num_generations,
                    inner_loop_size=self.config.inner_loop_size,
                    init=self.config.init,
                )
            )
            energy.run()
        else:
            raise ValueError(f"Unknown domain: {self.config.domain}")
        logger.info("Run complete for domain=%s mode=%s method=%s", domain, self.config.mode, self.config.method)
