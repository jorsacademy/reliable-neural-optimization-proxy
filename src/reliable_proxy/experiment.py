"""Frozen train/evaluation protocol for reliable optimization proxies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from reliable_proxy.benchmark import BenchmarkReport, run_benchmark
from reliable_proxy.dataset import build_dataset
from reliable_proxy.domain import DispatchRegime
from reliable_proxy.generator import generate_instances
from reliable_proxy.model import TrainingConfig
from reliable_proxy.proxy import ReliableOptimizationProxy
from reliable_proxy.training import TrainingReport, train_proxy

EVALUATION_REGIMES: tuple[DispatchRegime, ...] = (
    "in_distribution",
    "high_demand",
    "low_demand",
    "cost_shift",
    "capacity_shift",
    "combined_shift",
)


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    train_samples: int = 1500
    validation_samples: int = 300
    evaluation_samples: int = 200
    unit_count: int = 8
    hidden_dim: int = 64
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 2e-3
    patience: int = 15
    relative_gap_tolerance: float = 0.01
    absolute_gap_tolerance: float = 1e-6
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.train_samples <= 0 or self.validation_samples <= 0:
            raise ValueError("training and validation sample counts must be positive")
        if self.evaluation_samples <= 0 or self.unit_count < 2:
            raise ValueError("evaluation_samples must be positive and unit_count at least two")
        if self.relative_gap_tolerance < 0.0 or self.absolute_gap_tolerance < 0.0:
            raise ValueError("gap tolerances must be nonnegative")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    config: ResearchConfig
    training: TrainingReport
    scenarios: dict[str, BenchmarkReport]
    seed_ranges: dict[str, tuple[int, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "training": self.training.to_dict(),
            "scenarios": {
                name: report.to_dict() for name, report in self.scenarios.items()
            },
            "seed_ranges": {
                name: list(bounds) for name, bounds in self.seed_ranges.items()
            },
        }


def run_research_experiment(
    config: ResearchConfig | None = None,
) -> tuple[ReliableOptimizationProxy, ResearchReport]:
    config = config or ResearchConfig()
    train_seed = config.seed
    validation_seed = config.seed + 100_000
    evaluation_seed = config.seed + 200_000
    train_instances = generate_instances(
        config.train_samples,
        unit_count=config.unit_count,
        regime="in_distribution",
        seed=train_seed,
    )
    validation_instances = generate_instances(
        config.validation_samples,
        unit_count=config.unit_count,
        regime="in_distribution",
        seed=validation_seed,
    )
    train_dataset = build_dataset(train_instances)
    validation_dataset = build_dataset(validation_instances)
    proxy, training = train_proxy(
        train_dataset,
        validation_dataset,
        config=TrainingConfig(
            hidden_dim=config.hidden_dim,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=config.patience,
            seed=config.seed,
        ),
    )
    scenarios: dict[str, BenchmarkReport] = {}
    seed_ranges: dict[str, tuple[int, int]] = {
        "train": (train_seed, train_seed + config.train_samples - 1),
        "validation": (
            validation_seed,
            validation_seed + config.validation_samples - 1,
        ),
    }
    for index, regime in enumerate(EVALUATION_REGIMES):
        scenario_seed = evaluation_seed + index * 10_000
        instances = generate_instances(
            config.evaluation_samples,
            unit_count=config.unit_count,
            regime=regime,
            seed=scenario_seed,
        )
        scenarios[regime] = run_benchmark(
            instances,
            proxy,
            relative_gap_tolerance=config.relative_gap_tolerance,
            absolute_gap_tolerance=config.absolute_gap_tolerance,
        )
        seed_ranges[f"evaluation:{regime}"] = (
            scenario_seed,
            scenario_seed + config.evaluation_samples - 1,
        )
    return proxy, ResearchReport(
        config=config,
        training=training,
        scenarios=scenarios,
        seed_ranges=seed_ranges,
    )


def save_research_report(report: ResearchReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
