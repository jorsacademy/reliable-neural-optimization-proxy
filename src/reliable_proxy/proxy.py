"""Neural prediction, exact repair, dual certification, and safe fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from reliable_proxy.audit import FeasibilityAudit, audit_feasibility
from reliable_proxy.certificate import OptimalityCertificate, certify_candidate
from reliable_proxy.domain import DispatchInstance
from reliable_proxy.features import (
    FEATURE_SCHEMA_VERSION,
    decision_from_canonical,
    featurize,
    feature_names,
)
from reliable_proxy.model import NumpyMLPProxy
from reliable_proxy.oracle import ExactSolution, solve_exact
from reliable_proxy.projection import ProjectionResult, project_feasible


@dataclass(frozen=True, slots=True)
class ProxySolveResult:
    instance_name: str
    raw_decision: tuple[float, ...]
    raw_audit: FeasibilityAudit
    projected_decision: tuple[float, ...]
    projection: ProjectionResult
    certificate: OptimalityCertificate
    final_decision: tuple[float, ...]
    final_objective: float
    accepted_proxy: bool
    used_fallback: bool
    fallback_reason: str | None
    allowed_gap: float
    exact_solution: ExactSolution | None
    inference_seconds: float
    total_seconds: float


class ReliableOptimizationProxy:
    """A fixed-size neural proxy with deterministic reliability layers."""

    def __init__(self, model: NumpyMLPProxy, *, unit_count: int) -> None:
        expected_input = len(feature_names(unit_count))
        if model.input_dim != expected_input or model.output_dim != unit_count:
            raise ValueError("model dimensions are incompatible with unit_count")
        metadata_unit_count = model.metadata.get("unit_count")
        if metadata_unit_count is not None and int(metadata_unit_count) != unit_count:
            raise ValueError("checkpoint unit_count does not match the proxy")
        feature_schema = model.metadata.get("feature_schema_version")
        if feature_schema is not None and feature_schema != FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature schema is incompatible")
        self.model = model
        self.unit_count = unit_count

    @classmethod
    def load(cls, path: str | Path) -> ReliableOptimizationProxy:
        model = NumpyMLPProxy.load(path)
        unit_count_raw = model.metadata.get("unit_count", model.output_dim)
        return cls(model, unit_count=int(unit_count_raw))

    def predict_raw(self, instance: DispatchInstance) -> tuple[tuple[float, ...], float]:
        if instance.unit_count != self.unit_count:
            raise ValueError(
                f"proxy expects {self.unit_count} units but instance has {instance.unit_count}"
            )
        canonical = featurize(instance)
        started = time.perf_counter()
        prediction = self.model.predict(canonical.values[None, :])[0]
        runtime = time.perf_counter() - started
        restored = decision_from_canonical(prediction, canonical.order)
        return tuple(float(value) for value in restored), runtime

    def solve(
        self,
        instance: DispatchInstance,
        *,
        relative_gap_tolerance: float = 0.01,
        absolute_gap_tolerance: float = 1e-6,
    ) -> ProxySolveResult:
        if relative_gap_tolerance < 0.0 or absolute_gap_tolerance < 0.0:
            raise ValueError("gap tolerances must be nonnegative")
        started = time.perf_counter()
        raw, inference_seconds = self.predict_raw(instance)
        raw_audit = audit_feasibility(instance, raw)
        projection = project_feasible(instance, raw)
        certificate = certify_candidate(instance, projection.decision)
        allowed_gap = absolute_gap_tolerance + relative_gap_tolerance * max(
            1.0, abs(certificate.primal_objective)
        )
        accepted = certificate.valid and certificate.certified_gap <= allowed_gap
        if accepted:
            final_decision = projection.decision
            exact_solution = None
            fallback_reason = None
        else:
            exact_solution = solve_exact(instance)
            final_decision = exact_solution.decision
            fallback_reason = (
                "invalid feasibility certificate"
                if not certificate.valid
                else "certified gap exceeds the acceptance threshold"
            )
        final_objective = instance.objective(final_decision)
        if not audit_feasibility(instance, final_decision).feasible:
            raise RuntimeError("reliable proxy returned an infeasible final decision")
        return ProxySolveResult(
            instance_name=instance.name,
            raw_decision=raw,
            raw_audit=raw_audit,
            projected_decision=projection.decision,
            projection=projection,
            certificate=certificate,
            final_decision=final_decision,
            final_objective=final_objective,
            accepted_proxy=accepted,
            used_fallback=not accepted,
            fallback_reason=fallback_reason,
            allowed_gap=allowed_gap,
            exact_solution=exact_solution,
            inference_seconds=inference_seconds,
            total_seconds=time.perf_counter() - started,
        )
