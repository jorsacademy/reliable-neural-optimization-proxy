"""Deterministic synthetic parametric-dispatch instance generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reliable_proxy.domain import DispatchInstance, DispatchRegime


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    unit_count: int = 8
    regime: DispatchRegime = "in_distribution"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.unit_count < 2:
            raise ValueError("unit_count must be at least two")


@dataclass(frozen=True, slots=True)
class _Parameters:
    quadratic: tuple[float, float]
    linear: tuple[float, float]
    upper: tuple[float, float]
    lower_fraction: tuple[float, float]
    demand_ratio: tuple[float, float]


_REGIMES: dict[DispatchRegime, _Parameters] = {
    "in_distribution": _Parameters(
        (0.35, 2.25), (1.5, 8.0), (8.0, 24.0), (0.0, 0.10), (0.25, 0.75)
    ),
    "high_demand": _Parameters(
        (0.35, 2.25), (1.5, 8.0), (8.0, 24.0), (0.0, 0.10), (0.84, 0.97)
    ),
    "low_demand": _Parameters(
        (0.35, 2.25), (1.5, 8.0), (8.0, 24.0), (0.0, 0.10), (0.03, 0.18)
    ),
    "cost_shift": _Parameters(
        (1.6, 4.5), (7.0, 18.0), (8.0, 24.0), (0.0, 0.10), (0.25, 0.75)
    ),
    "capacity_shift": _Parameters(
        (0.35, 2.25), (1.5, 8.0), (22.0, 52.0), (0.0, 0.16), (0.25, 0.75)
    ),
    "combined_shift": _Parameters(
        (1.25, 4.0), (6.0, 16.0), (16.0, 42.0), (0.0, 0.16), (0.82, 0.96)
    ),
}


def generate_instance(config: GeneratorConfig | None = None, **overrides: object) -> DispatchInstance:
    """Generate one reproducible strictly feasible dispatch instance."""

    if config is not None and overrides:
        raise ValueError("provide either config or keyword overrides, not both")
    if config is None:
        config = GeneratorConfig(**overrides)  # type: ignore[arg-type]
    parameters = _REGIMES[config.regime]
    rng = np.random.default_rng(config.seed)
    quadratic = rng.uniform(*parameters.quadratic, size=config.unit_count)
    linear = rng.uniform(*parameters.linear, size=config.unit_count)
    upper = rng.uniform(*parameters.upper, size=config.unit_count)
    lower = upper * rng.uniform(*parameters.lower_fraction, size=config.unit_count)
    ratio = float(rng.uniform(*parameters.demand_ratio))
    demand = float(np.sum(lower) + ratio * np.sum(upper - lower))
    return DispatchInstance(
        name=f"dispatch-{config.regime}-n{config.unit_count}-seed{config.seed}",
        regime=config.regime,
        quadratic_costs=tuple(float(value) for value in quadratic),
        linear_costs=tuple(float(value) for value in linear),
        lower_bounds=tuple(float(value) for value in lower),
        upper_bounds=tuple(float(value) for value in upper),
        demand=demand,
    )


def generate_instances(
    count: int,
    *,
    unit_count: int,
    regime: DispatchRegime,
    seed: int,
) -> tuple[DispatchInstance, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    return tuple(
        generate_instance(GeneratorConfig(unit_count=unit_count, regime=regime, seed=seed + index))
        for index in range(count)
    )
