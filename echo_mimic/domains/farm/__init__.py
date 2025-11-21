"""Farm domain orchestration wrappers.

This module provides a thin, typed interface over the existing farm-specific
pipelines so that they can be invoked from a unified orchestration layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .farm_evo_strat import run as run_farm_local
from .graph_evo_strat import run as run_farm_global
from .nudge_evo_strat import run as run_farm_nudge

from echo_mimic.baselines.dspy import (
    run_dspy_farm as run_dspy_local,
    run_dspy_global,
    run_dspy_nudge,
)

from echo_mimic.baselines.autogen import AutoGenBaseline

logger = logging.getLogger(__name__)


@dataclass
class FarmDomainConfig:
    mode: str
    method: str
    agent_id: Optional[str]
    model: Optional[str]
    population_size: int
    num_generations: int
    inner_loop_size: int
    use_template: bool = False
    use_hint: bool = True
    halstead_metrics: bool = False
    init: bool = True


class FarmDomain:
    """Entry point for farm domain runs across methods."""

    def __init__(self, config: FarmDomainConfig):
        self.config = config
        self.autogen = AutoGenBaseline(domain="farm")

    def _resolve_agent_id(self) -> int:
        if self.config.agent_id is None:
            return 3
        try:
            return int(self.config.agent_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Farm domain requires an integer agent id, received {self.config.agent_id!r}"
            ) from exc

    def _echo_mimic_dispatch(self) -> None:
        target_farm_id = self._resolve_agent_id()
        if self.config.mode == "local":
            run_farm_local(
                population_size_value=self.config.population_size,
                num_generations_value=self.config.num_generations,
                inner_loop_size_value=self.config.inner_loop_size,
                agent_id=target_farm_id,
                init_value=self.config.init,
                use_template_value=self.config.use_template,
                halstead_metrics_value=self.config.halstead_metrics,
                model_name=self.config.model,
            )
        elif self.config.mode == "global":
            run_farm_global(
                population_size_value=self.config.population_size,
                num_generations_value=self.config.num_generations,
                agent_id=target_farm_id,
                init_value=self.config.init,
                use_hint_value=self.config.use_hint,
                use_template_value=self.config.use_template,
                halstead_metrics_value=self.config.halstead_metrics,
                model_name=self.config.model,
            )
        elif self.config.mode == "nudge":
            run_farm_nudge(
                population_size_value=self.config.population_size,
                num_generations_value=self.config.num_generations,
                inner_loop_size_value=self.config.inner_loop_size,
                agent_id=target_farm_id,
                init_value=self.config.init,
                model_name=self.config.model,
            )
        else:
            raise ValueError(f"Unsupported farm mode: {self.config.mode}")

    def _dspy_dispatch(self) -> None:
        model_name = self.config.model
        if self.config.mode == "local":
            run_dspy_local(model_name=model_name)
        elif self.config.mode == "global":
            run_dspy_global(model_name=model_name)
        elif self.config.mode == "nudge":
            run_dspy_nudge(model_name=model_name)
        else:
            raise ValueError(f"Unsupported DSPy farm mode: {self.config.mode}")

    def _autogen_dispatch(self) -> None:
        self.autogen.run(
            mode=self.config.mode,
            agent_id=self.config.agent_id,
            model=self.config.model,
            data_hint="data/farm/farm_*/",
        )

    def run(self) -> None:
        handlers: Dict[str, Callable[[], None]] = {
            "echo_mimic": self._echo_mimic_dispatch,
            "dspy": self._dspy_dispatch,
            "autogen": self._autogen_dispatch,
        }
        method = self.config.method.lower()
        if method not in handlers:
            raise ValueError(f"Unsupported farm method: {self.config.method}")
        logger.info("Running farm domain with method=%s mode=%s agent=%s model=%s", method, self.config.mode, self.config.agent_id, self.config.model)
        handlers[method]()
