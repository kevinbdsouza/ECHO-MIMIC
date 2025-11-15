"""Unified entry point to launch different ECHO-MIMIC experiments."""

import argparse
from typing import List, Optional


def run_baseline_local(_args: argparse.Namespace) -> None:
    from dspy_baseline_farm import main as baseline_main

    baseline_main()


def run_baseline_global(_args: argparse.Namespace) -> None:
    from dspy_baseline_global import main as baseline_main

    baseline_main()


def run_baseline_nudge(_args: argparse.Namespace) -> None:
    from dspy_baseline_nudge import main as baseline_main

    baseline_main()


def run_echo_local(args: argparse.Namespace) -> None:
    from farm_evo_strat import run as farm_run

    farm_run(
        population_size_value=args.population_size,
        num_generations_value=args.num_generations,
        inner_loop_size_value=args.inner_loop_size,
        farm_ids=args.farm_ids,
        init_value=not args.no_init,
        use_template_value=args.use_template,
        halstead_metrics_value=args.halstead_metrics,
    )


def run_echo_global(args: argparse.Namespace) -> None:
    from graph_evo_strat import run as graph_run

    graph_run(
        population_size_value=args.population_size,
        num_generations_value=args.num_generations,
        farm_ids=args.farm_ids,
        init_value=not args.no_init,
        use_hint_value=not args.no_hint,
        use_template_value=args.use_template,
        halstead_metrics_value=args.halstead_metrics,
    )


def run_mimic(args: argparse.Namespace) -> None:
    from nudge_evo_strat import run as nudge_run

    nudge_run(
        population_size_value=args.population_size,
        num_generations_value=args.num_generations,
        inner_loop_size_value=args.inner_loop_size,
        farm_ids=args.farm_ids,
        init_value=not args.no_init,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ECHO-MIMIC experiment workflows from a single entry point.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_local = subparsers.add_parser(
        "baseline-local",
        help="Run the DSPy baseline for local farm interventions.",
    )
    baseline_local.set_defaults(func=run_baseline_local)

    baseline_global = subparsers.add_parser(
        "baseline-global",
        help="Run the DSPy baseline for global connectivity interventions.",
    )
    baseline_global.set_defaults(func=run_baseline_global)

    baseline_nudge = subparsers.add_parser(
        "baseline-nudge",
        help="Run the DSPy baseline for policy nudging.",
    )
    baseline_nudge.set_defaults(func=run_baseline_nudge)

    echo_local = subparsers.add_parser(
        "echo-local",
        help="Run the ECHO evolutionary strategy for local farm heuristics.",
    )
    echo_local.add_argument("--population-size", type=int, default=25)
    echo_local.add_argument("--num-generations", type=int, default=25)
    echo_local.add_argument("--inner-loop-size", type=int, default=25)
    echo_local.add_argument("--farm-ids", type=int, nargs="+", default=[3])
    echo_local.add_argument("--no-init", action="store_true")
    echo_local.add_argument("--use-template", action="store_true")
    echo_local.add_argument("--halstead-metrics", action="store_true")
    echo_local.set_defaults(func=run_echo_local)

    echo_global = subparsers.add_parser(
        "echo-global",
        help="Run the ECHO evolutionary strategy for global connectivity heuristics.",
    )
    echo_global.add_argument("--population-size", type=int, default=25)
    echo_global.add_argument("--num-generations", type=int, default=25)
    echo_global.add_argument("--farm-ids", type=int, nargs="+", default=[3])
    echo_global.add_argument("--no-init", action="store_true")
    echo_global.add_argument("--no-hint", action="store_true")
    echo_global.add_argument("--use-template", action="store_true")
    echo_global.add_argument("--halstead-metrics", action="store_true")
    echo_global.set_defaults(func=run_echo_global)

    mimic = subparsers.add_parser(
        "mimic",
        help="Run the policy nudge evolutionary strategy (ECHO-MIMIC).",
    )
    mimic.add_argument("--population-size", type=int, default=25)
    mimic.add_argument("--num-generations", type=int, default=10)
    mimic.add_argument("--inner-loop-size", type=int, default=10)
    mimic.add_argument("--farm-ids", type=int, nargs="+", default=[3])
    mimic.add_argument("--no-init", action="store_true")
    mimic.set_defaults(func=run_mimic)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
