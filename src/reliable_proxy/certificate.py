"""Dual lower bounds and deterministic suboptimality certificates."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from reliable_proxy.audit import KKTAudit, audit_kkt
from reliable_proxy.domain import DispatchInstance
from reliable_proxy.oracle import dual_value


@dataclass(frozen=True, slots=True)
class OptimalityCertificate:
    valid: bool
    multiplier: float
    primal_objective: float
    dual_lower_bound: float
    certified_gap: float
    certified_gap_percent: float
    kkt_audit: KKTAudit
    runtime_seconds: float


def estimate_multiplier(
    instance: DispatchInstance,
    decision: tuple[float, ...] | np.ndarray,
    *,
    active_tolerance: float = 1e-7,
) -> float:
    """Estimate a dual multiplier from candidate marginal costs."""

    values = np.asarray(decision, dtype=float)
    if values.shape != (instance.unit_count,):
        raise ValueError("decision has the wrong shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("decision must be finite")
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    candidate_values = -(quadratic * values + linear)
    interior = (values > lower + active_tolerance) & (values < upper - active_tolerance)
    if np.any(interior):
        return float(np.median(candidate_values[interior]))
    at_lower = values <= lower + active_tolerance
    at_upper = values >= upper - active_tolerance
    lower_requirement = float(np.max(candidate_values[at_lower])) if np.any(at_lower) else -np.inf
    upper_requirement = float(np.min(candidate_values[at_upper])) if np.any(at_upper) else np.inf
    if np.isfinite(lower_requirement) and np.isfinite(upper_requirement):
        return 0.5 * (lower_requirement + upper_requirement)
    if np.isfinite(lower_requirement):
        return lower_requirement
    if np.isfinite(upper_requirement):
        return upper_requirement
    return float(np.median(candidate_values))


def certify_candidate(
    instance: DispatchInstance,
    decision: tuple[float, ...] | np.ndarray,
    *,
    multiplier: float | None = None,
    feasibility_tolerance: float = 1e-8,
) -> OptimalityCertificate:
    """Use weak duality to upper-bound candidate suboptimality."""

    started = time.perf_counter()
    values = np.asarray(decision, dtype=float)
    if values.shape != (instance.unit_count,):
        raise ValueError("decision has the wrong shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("decision must be finite")
    candidate_multiplier = estimate_multiplier(instance, values) if multiplier is None else float(multiplier)
    if not np.isfinite(candidate_multiplier):
        raise ValueError("multiplier must be finite")
    kkt = audit_kkt(instance, values, candidate_multiplier, feasibility_tolerance=feasibility_tolerance)
    decision_tuple = tuple(float(value) for value in values)
    primal = instance.objective(decision_tuple)
    lower_bound = dual_value(instance, candidate_multiplier)
    gap = primal - lower_bound
    numerical_tolerance = 1e-9 * max(1.0, abs(primal), abs(lower_bound))
    if not kkt.feasibility.feasible:
        gap = float("inf")
    elif gap < -numerical_tolerance:
        raise RuntimeError("computed dual value exceeds the feasible candidate objective")
    else:
        gap = max(0.0, gap)
    return OptimalityCertificate(
        valid=kkt.feasibility.feasible,
        multiplier=candidate_multiplier,
        primal_objective=primal,
        dual_lower_bound=lower_bound,
        certified_gap=gap,
        certified_gap_percent=100.0 * gap / max(1.0, abs(primal)),
        kkt_audit=kkt,
        runtime_seconds=time.perf_counter() - started,
    )
