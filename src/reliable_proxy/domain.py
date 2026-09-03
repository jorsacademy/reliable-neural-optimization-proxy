"""Typed parametric economic-dispatch instances and JSON serialization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

DispatchRegime = Literal[
    "in_distribution",
    "high_demand",
    "low_demand",
    "cost_shift",
    "capacity_shift",
    "combined_shift",
]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _number_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class DispatchInstance:
    """A strictly convex separable dispatch problem.

    The objective is ``sum_i 0.5*a_i*x_i**2 + b_i*x_i`` subject to
    ``sum_i x_i = demand`` and ``lower_i <= x_i <= upper_i``.
    """

    name: str
    quadratic_costs: tuple[float, ...]
    linear_costs: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    demand: float
    regime: DispatchRegime = "in_distribution"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be nonempty")
        size = len(self.quadratic_costs)
        if size < 2:
            raise ValueError("at least two production units are required")
        if not (
            len(self.linear_costs)
            == len(self.lower_bounds)
            == len(self.upper_bounds)
            == size
        ):
            raise ValueError("all coefficient vectors must have the same length")
        values = (
            *self.quadratic_costs,
            *self.linear_costs,
            *self.lower_bounds,
            *self.upper_bounds,
            self.demand,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all instance values must be finite")
        if any(value <= 0.0 for value in self.quadratic_costs):
            raise ValueError("quadratic_costs must be strictly positive")
        if any(value < 0.0 for value in self.linear_costs):
            raise ValueError("linear_costs must be nonnegative")
        for index, (lower, upper) in enumerate(
            zip(self.lower_bounds, self.upper_bounds, strict=True)
        ):
            if lower < 0.0:
                raise ValueError(f"lower_bounds[{index}] must be nonnegative")
            if upper <= lower:
                raise ValueError(f"upper_bounds[{index}] must exceed its lower bound")
        tolerance = 1e-10 * max(1.0, abs(self.demand), abs(self.upper_total))
        if self.demand < self.lower_total - tolerance or self.demand > self.upper_total + tolerance:
            raise ValueError(
                "demand must lie between the aggregate lower and upper production bounds"
            )

    @property
    def unit_count(self) -> int:
        return len(self.quadratic_costs)

    @property
    def lower_total(self) -> float:
        return float(sum(self.lower_bounds))

    @property
    def upper_total(self) -> float:
        return float(sum(self.upper_bounds))

    @property
    def flexible_capacity(self) -> float:
        return self.upper_total - self.lower_total

    @property
    def demand_ratio(self) -> float:
        return (self.demand - self.lower_total) / max(self.flexible_capacity, 1e-12)

    def objective(self, decision: tuple[float, ...]) -> float:
        if len(decision) != self.unit_count:
            raise ValueError("decision has the wrong length")
        if not all(math.isfinite(value) for value in decision):
            raise ValueError("decision values must be finite")
        return float(
            sum(
                0.5 * quadratic * value * value + linear * value
                for quadratic, linear, value in zip(
                    self.quadratic_costs,
                    self.linear_costs,
                    decision,
                    strict=True,
                )
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "regime": self.regime,
            "quadratic_costs": list(self.quadratic_costs),
            "linear_costs": list(self.linear_costs),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "demand": self.demand,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DispatchInstance:
        raw_regime = payload.get("regime", "in_distribution")
        allowed = {
            "in_distribution",
            "high_demand",
            "low_demand",
            "cost_shift",
            "capacity_shift",
            "combined_shift",
        }
        if not isinstance(raw_regime, str) or raw_regime not in allowed:
            raise ValueError(f"unknown regime: {raw_regime!r}")
        raw_name = payload.get("name")
        if not isinstance(raw_name, str):
            raise ValueError("name must be a string")
        return cls(
            name=raw_name,
            regime=cast(DispatchRegime, raw_regime),
            quadratic_costs=_number_tuple(payload.get("quadratic_costs"), "quadratic_costs"),
            linear_costs=_number_tuple(payload.get("linear_costs"), "linear_costs"),
            lower_bounds=_number_tuple(payload.get("lower_bounds"), "lower_bounds"),
            upper_bounds=_number_tuple(payload.get("upper_bounds"), "upper_bounds"),
            demand=_number(payload.get("demand"), "demand"),
        )


def save_instance(instance: DispatchInstance, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(instance.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_instance(path: str | Path) -> DispatchInstance:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("instance JSON must contain an object")
    return DispatchInstance.from_dict(cast(dict[str, object], payload))
