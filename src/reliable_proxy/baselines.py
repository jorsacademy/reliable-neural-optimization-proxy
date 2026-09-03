"""Transparent non-neural dispatch baselines."""

from __future__ import annotations

import numpy as np

from reliable_proxy.domain import DispatchInstance


def proportional_dispatch(instance: DispatchInstance) -> tuple[float, ...]:
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    decision = lower + instance.demand_ratio * (upper - lower)
    return tuple(float(value) for value in decision)


def merit_order_dispatch(instance: DispatchInstance) -> tuple[float, ...]:
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    decision = lower.copy()
    remaining = instance.demand - float(np.sum(lower))
    marginal = quadratic * lower + linear
    for index in np.argsort(marginal, kind="stable"):
        addition = min(remaining, upper[index] - lower[index])
        decision[index] += addition
        remaining -= addition
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise RuntimeError("merit-order heuristic failed to meet demand")
    return tuple(float(value) for value in decision)
