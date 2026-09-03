"""Canonical permutation-stable features for the neural proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reliable_proxy.domain import DispatchInstance

LOCAL_FEATURES = (
    "quadratic_cost",
    "linear_cost",
    "lower_bound",
    "upper_bound",
    "capacity_span",
    "lower_fraction",
)
GLOBAL_FEATURES = (
    "demand",
    "aggregate_lower",
    "aggregate_upper",
    "flexible_capacity",
    "demand_ratio",
    "mean_quadratic_cost",
    "std_quadratic_cost",
    "mean_linear_cost",
    "std_linear_cost",
)
FEATURE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class CanonicalFeatures:
    values: np.ndarray
    order: tuple[int, ...]


def canonical_order(instance: DispatchInstance) -> tuple[int, ...]:
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    index = np.arange(instance.unit_count)
    order = np.lexsort((index, upper, lower, quadratic, linear))
    return tuple(int(value) for value in order)


def feature_names(unit_count: int) -> tuple[str, ...]:
    if unit_count < 2:
        raise ValueError("unit_count must be at least two")
    local = tuple(
        f"unit_{unit_index}_{feature}"
        for unit_index in range(unit_count)
        for feature in LOCAL_FEATURES
    )
    return (*local, *GLOBAL_FEATURES)


def featurize(instance: DispatchInstance) -> CanonicalFeatures:
    order_tuple = canonical_order(instance)
    order = np.asarray(order_tuple, dtype=int)
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)[order]
    linear = np.asarray(instance.linear_costs, dtype=float)[order]
    lower = np.asarray(instance.lower_bounds, dtype=float)[order]
    upper = np.asarray(instance.upper_bounds, dtype=float)[order]
    span = upper - lower
    lower_fraction = lower / np.maximum(upper, 1e-12)
    local = np.column_stack((quadratic, linear, lower, upper, span, lower_fraction)).reshape(-1)
    global_values = np.asarray(
        [
            instance.demand,
            instance.lower_total,
            instance.upper_total,
            instance.flexible_capacity,
            instance.demand_ratio,
            float(np.mean(quadratic)),
            float(np.std(quadratic)),
            float(np.mean(linear)),
            float(np.std(linear)),
        ],
        dtype=float,
    )
    return CanonicalFeatures(values=np.concatenate((local, global_values)), order=order_tuple)


def decision_to_canonical(
    decision: tuple[float, ...] | np.ndarray,
    order: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(decision, dtype=float)
    if values.shape != (len(order),):
        raise ValueError("decision and order have incompatible shapes")
    return values[np.asarray(order, dtype=int)]


def decision_from_canonical(
    canonical_decision: tuple[float, ...] | np.ndarray,
    order: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(canonical_decision, dtype=float)
    if values.shape != (len(order),):
        raise ValueError("canonical decision and order have incompatible shapes")
    restored = np.empty_like(values)
    restored[np.asarray(order, dtype=int)] = values
    return restored
