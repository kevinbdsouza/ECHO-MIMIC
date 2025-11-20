"""Unified entry point for all domains and methods."""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from echo_mimic.orchestrator import Orchestrator, RunConfig

logging.basicConfig(level=logging.INFO)


MODES = {"local", "global", "nudge"}
METHODS = {"echo_mimic", "dspy", "autogen"}
DOMAINS = {"farm", "energy"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ECHO-MIMIC pipelines in a modular fashion.")
    parser.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--method", choices=sorted(METHODS), default="echo_mimic")
    parser.add_argument("--agent-id", dest="agent_id", default=None)
    parser.add_argument("--model", dest="model", default=None, help="Model name for Gemini/OpenAI")
    parser.add_argument("--population-size", type=int, default=25)
    parser.add_argument("--num-generations", type=int, default=25)
    parser.add_argument("--inner-loop-size", type=int, default=10)
    parser.add_argument("--farm-ids", type=int, nargs="+", default=None)
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--no-hint", action="store_true")
    parser.add_argument("--halstead-metrics", action="store_true")
    parser.add_argument("--no-init", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        domain=args.domain,
        mode=args.mode,
        method=args.method,
        agent_id=args.agent_id,
        model=args.model,
        population_size=args.population_size,
        num_generations=args.num_generations,
        inner_loop_size=args.inner_loop_size,
        farm_ids=args.farm_ids,
        use_template=args.use_template,
        use_hint=not args.no_hint,
        halstead_metrics=args.halstead_metrics,
        init=not args.no_init,
    )
    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
