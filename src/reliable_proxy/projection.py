"""Exact Euclidean feasibility repair."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from reliable_proxy.audit import FeasibilityAudit, audit_feasibility
from reliable_proxy.domain import DispatchInstance


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    decision: tuple[float, ...]
    multiplier: float
    iterations: int
    correction_norm: float
    audit: FeasibilityAudit
    runtime_seconds: float


def project_feasible(
    instance: DispatchInstance,
    raw_decision: tuple[float, ...] | np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> ProjectionResult:
    """Project onto ``sum(x)=demand`` and the unit box constraints."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    raw = np.asarray(raw_decision, dtype=float)
    if raw.shape != (instance.unit_count,):
        raise ValueError("raw_decision has the wrong shape")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_decision must be finite")
    lower = np.asarray(instance.lower_bounds, dtype=float)
    upper = np.asarray(instance.upper_bounds, dtype=float)
    started = time.perf_counter()
    scale = max(1.0, abs(instance.demand), instance.upper_total)
    edge_tolerance = tolerance * scale
    if abs(instance.demand - instance.lower_total) <= edge_tolerance:
        projected = lower.copy()
        audit = audit_feasibility(instance, projected, tolerance=max(1e-10, tolerance * 100.0))
        return ProjectionResult(
            decision=tuple(float(value) for value in projected),
            multiplier=float(np.max(raw - lower)),
            iterations=0,
            correction_norm=float(np.linalg.norm(projected - raw)),
            audit=audit,
            runtime_seconds=time.perf_counter() - started,
        )
    if abs(instance.demand - instance.upper_total) <= edge_tolerance:
        projected = upper.copy()
        audit = audit_feasibility(instance, projected, tolerance=max(1e-10, tolerance * 100.0))
        return ProjectionResult(
            decision=tuple(float(value) for value in projected),
            multiplier=float(np.min(raw - upper)),
            iterations=0,
            correction_norm=float(np.linalg.norm(projected - raw)),
            audit=audit,
            runtime_seconds=time.perf_counter() - started,
        )
    margin = max(1.0, float(np.max(np.abs(raw))), float(np.max(np.abs(upper))))
    low = float(np.min(raw - upper) - margin)
    high = float(np.max(raw - lower) + margin)
    projected = np.clip(raw, lower, upper)
    multiplier = 0.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        multiplier = 0.5 * (low + high)
        projected = np.clip(raw - multiplier, lower, upper)
        residual = float(np.sum(projected)) - instance.demand
        if abs(residual) <= tolerance * scale:
            break
        if residual > 0.0:
            low = multiplier
        else:
            high = multiplier
    else:
        raise RuntimeError("feasibility projection did not converge")
    audit = audit_feasibility(instance, projected, tolerance=max(1e-10, tolerance * 100.0))
    if not audit.feasible:
        raise RuntimeError(f"projection failed feasibility audit: {audit.max_violation:.3e}")
    return ProjectionResult(
        decision=tuple(float(value) for value in projected),
        multiplier=float(multiplier),
        iterations=iterations,
        correction_norm=float(np.linalg.norm(projected - raw)),
        audit=audit,
        runtime_seconds=time.perf_counter() - started,
    )
