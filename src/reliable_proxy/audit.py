"""Independent feasibility and KKT audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reliable_proxy.domain import DispatchInstance


@dataclass(frozen=True, slots=True)
class FeasibilityAudit:
    balance_violation: float
    lower_bound_violation: float
    upper_bound_violation: float
    max_violation: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class KKTAudit:
    feasibility: FeasibilityAudit
    stationarity_residual: float
    multiplier: float


def audit_feasibility(
    instance: DispatchInstance,
    decision: tuple[float, ...] | np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> FeasibilityAudit:
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    values = np.asarray(decision, dtype=float)
    if values.shape != (instance.unit_count,):
        raise ValueError("decision has the wrong shape")
    if not np.all(np.isfinite(values)):
        return FeasibilityAudit(float("inf"), float("inf"), float("inf"), float("inf"), False)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    balance = abs(float(np.sum(values)) - instance.demand)
    lower_violation = float(np.max(np.maximum(lower - values, 0.0)))
    upper_violation = float(np.max(np.maximum(values - upper, 0.0)))
    maximum = max(balance, lower_violation, upper_violation)
    scale = max(1.0, abs(instance.demand), instance.upper_total)
    return FeasibilityAudit(
        balance_violation=balance,
        lower_bound_violation=lower_violation,
        upper_bound_violation=upper_violation,
        max_violation=maximum,
        feasible=maximum <= tolerance * scale,
    )


def audit_kkt(
    instance: DispatchInstance,
    decision: tuple[float, ...] | np.ndarray,
    multiplier: float,
    *,
    active_tolerance: float = 1e-8,
    feasibility_tolerance: float = 1e-8,
) -> KKTAudit:
    values = np.asarray(decision, dtype=float)
    if values.shape != (instance.unit_count,):
        raise ValueError("decision has the wrong shape")
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    stationarity = quadratic * values + linear + multiplier
    residuals = np.zeros_like(stationarity)
    at_lower = values <= lower + active_tolerance
    at_upper = values >= upper - active_tolerance
    interior = ~(at_lower | at_upper)
    residuals[interior] = np.abs(stationarity[interior])
    residuals[at_lower] = np.maximum(-stationarity[at_lower], 0.0)
    residuals[at_upper] = np.maximum(stationarity[at_upper], 0.0)
    return KKTAudit(
        feasibility=audit_feasibility(instance, values, tolerance=feasibility_tolerance),
        stationarity_residual=float(np.max(residuals)),
        multiplier=float(multiplier),
    )
