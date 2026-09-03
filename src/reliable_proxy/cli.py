"""Command-line interface for training and auditing optimization proxies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from reliable_proxy.benchmark import run_benchmark, save_report_csv, save_report_json
from reliable_proxy.dataset import build_dataset, load_dataset, save_dataset
from reliable_proxy.domain import DispatchInstance, DispatchRegime, load_instance, save_instance
from reliable_proxy.experiment import ResearchConfig, run_research_experiment, save_research_report
from reliable_proxy.generator import GeneratorConfig, generate_instance, generate_instances
from reliable_proxy.model import TrainingConfig
from reliable_proxy.proxy import ReliableOptimizationProxy
from reliable_proxy.training import train_proxy

REGIMES = (
    "in_distribution",
    "high_demand",
    "low_demand",
    "cost_shift",
    "capacity_shift",
    "combined_shift",
)


def _add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--units", type=int, default=8)
    parser.add_argument("--regime", choices=REGIMES, default="in_distribution")
    parser.add_argument("--seed", type=int, default=0)


def _instance_from_args(args: argparse.Namespace) -> DispatchInstance:
    return generate_instance(
        GeneratorConfig(
            unit_count=args.units,
            regime=args.regime,
            seed=args.seed,
        )
    )


def _write_or_print(payload: dict[str, object], output: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reliable-proxy",
        description="Neural optimization proxy with exact repair, dual certificates, and fallback.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write one deterministic dispatch instance")
    _add_instance_arguments(generate)
    generate.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="build an oracle-labelled proxy dataset")
    _add_instance_arguments(collect)
    collect.add_argument("--samples", type=int, default=1000)
    collect.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="train a neural proxy from two datasets")
    train.add_argument("train_dataset", type=Path)
    train.add_argument("validation_dataset", type=Path)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--output-report", type=Path)

    solve = subparsers.add_parser("solve", help="solve one instance with the certified proxy")
    solve.add_argument("--input", type=Path)
    _add_instance_arguments(solve)
    solve.add_argument("--checkpoint", type=Path, required=True)
    solve.add_argument("--relative-gap-tolerance", type=float, default=0.01)
    solve.add_argument("--absolute-gap-tolerance", type=float, default=1e-6)
    solve.add_argument("--include-vectors", action="store_true")
    solve.add_argument("--output", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare raw, repaired, certified, heuristic, and exact methods"
    )
    _add_instance_arguments(benchmark)
    benchmark.add_argument("--samples", type=int, default=100)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--relative-gap-tolerance", type=float, default=0.01)
    benchmark.add_argument("--absolute-gap-tolerance", type=float, default=1e-6)
    benchmark.add_argument("--output-json", type=Path)
    benchmark.add_argument("--output-csv", type=Path)

    research = subparsers.add_parser(
        "research", help="run the frozen in-distribution and shift protocol"
    )
    research.add_argument("--train-samples", type=int, default=1500)
    research.add_argument("--validation-samples", type=int, default=300)
    research.add_argument("--evaluation-samples", type=int, default=200)
    research.add_argument("--units", type=int, default=8)
    research.add_argument("--hidden-dim", type=int, default=64)
    research.add_argument("--epochs", type=int, default=100)
    research.add_argument("--batch-size", type=int, default=128)
    research.add_argument("--learning-rate", type=float, default=2e-3)
    research.add_argument("--patience", type=int, default=15)
    research.add_argument("--relative-gap-tolerance", type=float, default=0.01)
    research.add_argument("--absolute-gap-tolerance", type=float, default=1e-6)
    research.add_argument("--seed", type=int, default=2026)
    research.add_argument("--checkpoint", type=Path, required=True)
    research.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            instance = _instance_from_args(args)
            save_instance(instance, args.output)
            _write_or_print({"output": str(args.output), "instance": instance.to_dict()})
            return 0

        if args.command == "collect":
            if args.samples <= 0:
                raise ValueError("--samples must be positive")
            instances = generate_instances(
                args.samples,
                unit_count=args.units,
                regime=args.regime,
                seed=args.seed,
            )
            dataset = build_dataset(instances)
            save_dataset(dataset, args.output)
            _write_or_print({"output": str(args.output), **dataset.summary()})
            return 0

        if args.command == "train":
            train_dataset = load_dataset(args.train_dataset)
            validation_dataset = load_dataset(args.validation_dataset)
            proxy, report = train_proxy(
                train_dataset,
                validation_dataset,
                config=TrainingConfig(
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    seed=args.seed,
                ),
            )
            proxy.model.save(args.checkpoint)
            payload = {"checkpoint": str(args.checkpoint), **report.to_dict()}
            _write_or_print(payload, args.output_report)
            return 0

        if args.command == "solve":
            instance = load_instance(args.input) if args.input else _instance_from_args(args)
            proxy = ReliableOptimizationProxy.load(args.checkpoint)
            result = proxy.solve(
                instance,
                relative_gap_tolerance=args.relative_gap_tolerance,
                absolute_gap_tolerance=args.absolute_gap_tolerance,
            )
            payload = asdict(result)
            if not args.include_vectors:
                for key in ("raw_decision", "projected_decision", "final_decision"):
                    payload.pop(key, None)
            _write_or_print(payload, args.output)
            return 0

        if args.command == "benchmark":
            if args.samples <= 0:
                raise ValueError("--samples must be positive")
            proxy = ReliableOptimizationProxy.load(args.checkpoint)
            instances = generate_instances(
                args.samples,
                unit_count=args.units,
                regime=args.regime,
                seed=args.seed,
            )
            report = run_benchmark(
                instances,
                proxy,
                relative_gap_tolerance=args.relative_gap_tolerance,
                absolute_gap_tolerance=args.absolute_gap_tolerance,
            )
            if args.output_json:
                save_report_json(report, args.output_json)
            if args.output_csv:
                save_report_csv(report, args.output_csv)
            _write_or_print(report.to_dict())
            return 0

        config = ResearchConfig(
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            evaluation_samples=args.evaluation_samples,
            unit_count=args.units,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            relative_gap_tolerance=args.relative_gap_tolerance,
            absolute_gap_tolerance=args.absolute_gap_tolerance,
            seed=args.seed,
        )
        proxy, report = run_research_experiment(config)
        proxy.model.save(args.checkpoint)
        save_research_report(report, args.output_report)
        _write_or_print(
            {
                "checkpoint": str(args.checkpoint),
                "report": str(args.output_report),
                "validation_rmse": report.training.validation_rmse,
            }
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
