"""Exact primal-dual oracle for separable convex dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from reliable_proxy.audit import audit_kkt
from reliable_proxy.domain import DispatchInstance


@dataclass(frozen=True, slots=True)
class ExactSolution:
    decision: tuple[float, ...]
    objective: float
    dual_multiplier: float
    dual_value: float
    duality_gap: float
    stationarity_residual: float
    balance_violation: float
    iterations: int
    runtime_seconds: float


def box_minimizer(instance: DispatchInstance, multiplier: float) -> np.ndarray:
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    return np.clip(-(linear + multiplier) / quadratic, lower, upper)


def dual_value(instance: DispatchInstance, multiplier: float) -> float:
    minimizer = box_minimizer(instance, multiplier)
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    terms = 0.5 * quadratic * minimizer * minimizer + (linear + multiplier) * minimizer
    return float(np.sum(terms) - multiplier * instance.demand)


def _bracket(instance: DispatchInstance) -> tuple[float, float]:
    quadratic = np.asarray(instance.quadratic_costs, dtype=float)
    linear = np.asarray(instance.linear_costs, dtype=float)
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    low_break = -linear - quadratic * upper
    high_break = -linear - quadratic * lower
    margin = max(1.0, float(np.max(np.abs(np.concatenate((low_break, high_break))))))
    return float(np.min(low_break) - margin), float(np.max(high_break) + margin)


def solve_exact(
    instance: DispatchInstance,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> ExactSolution:
    """Solve the unique optimum through a monotone KKT bisection."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    started = time.perf_counter()
    scale = max(1.0, abs(instance.demand), instance.upper_total)
    edge_tolerance = tolerance * scale
    if abs(instance.demand - instance.lower_total) <= edge_tolerance:
        decision = np.asarray(instance.lower_bounds, dtype=float)
        quadratic = np.asarray(instance.quadratic_costs, dtype=float)
        linear = np.asarray(instance.linear_costs, dtype=float)
        multiplier = float(np.max(-(quadratic * decision + linear)) + 1.0)
        decision_tuple = tuple(float(value) for value in decision)
        objective = instance.objective(decision_tuple)
        lower_bound = dual_value(instance, multiplier)
        audit = audit_kkt(instance, decision, multiplier)
        return ExactSolution(
            decision=decision_tuple,
            objective=objective,
            dual_multiplier=multiplier,
            dual_value=lower_bound,
            duality_gap=max(0.0, objective - lower_bound),
            stationarity_residual=audit.stationarity_residual,
            balance_violation=audit.feasibility.balance_violation,
            iterations=0,
            runtime_seconds=time.perf_counter() - started,
        )
    if abs(instance.demand - instance.upper_total) <= edge_tolerance:
        decision = np.asarray(instance.upper_bounds, dtype=float)
        quadratic = np.asarray(instance.quadratic_costs, dtype=float)
        linear = np.asarray(instance.linear_costs, dtype=float)
        multiplier = float(np.min(-(quadratic * decision + linear)) - 1.0)
        decision_tuple = tuple(float(value) for value in decision)
        objective = instance.objective(decision_tuple)
        lower_bound = dual_value(instance, multiplier)
        audit = audit_kkt(instance, decision, multiplier)
        return ExactSolution(
            decision=decision_tuple,
            objective=objective,
            dual_multiplier=multiplier,
            dual_value=lower_bound,
            duality_gap=max(0.0, objective - lower_bound),
            stationarity_residual=audit.stationarity_residual,
            balance_violation=audit.feasibility.balance_violation,
            iterations=0,
            runtime_seconds=time.perf_counter() - started,
        )
    low, high = _bracket(instance)
    if float(np.sum(box_minimizer(instance, low))) < instance.demand - tolerance:
        raise RuntimeError("failed to bracket the KKT multiplier from below")
    if float(np.sum(box_minimizer(instance, high))) > instance.demand + tolerance:
        raise RuntimeError("failed to bracket the KKT multiplier from above")
    multiplier = 0.5 * (low + high)
    decision = box_minimizer(instance, multiplier)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        multiplier = 0.5 * (low + high)
        decision = box_minimizer(instance, multiplier)
        balance = float(np.sum(decision)) - instance.demand
        if abs(balance) <= tolerance * scale:
            break
        if balance > 0.0:
            low = multiplier
        else:
            high = multiplier
    else:
        raise RuntimeError("exact KKT bisection did not converge")
    decision_tuple = tuple(float(value) for value in decision)
    objective = instance.objective(decision_tuple)
    lower_bound = dual_value(instance, multiplier)
    gap = objective - lower_bound
    if gap < -1e-8 * max(1.0, abs(objective)):
        raise RuntimeError("dual lower bound exceeds the primal objective")
    audit = audit_kkt(instance, decision, multiplier)
    if not audit.feasibility.feasible:
        raise RuntimeError("exact oracle returned an infeasible decision")
    return ExactSolution(
        decision=decision_tuple,
        objective=objective,
        dual_multiplier=float(multiplier),
        dual_value=lower_bound,
        duality_gap=max(0.0, gap),
        stationarity_residual=audit.stationarity_residual,
        balance_violation=audit.feasibility.balance_violation,
        iterations=iterations,
        runtime_seconds=time.perf_counter() - started,
    )
