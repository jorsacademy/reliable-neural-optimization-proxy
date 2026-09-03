"""Solver-grounded evaluation of raw, repaired, and certified proxies."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from reliable_proxy.audit import audit_feasibility
from reliable_proxy.baselines import merit_order_dispatch, proportional_dispatch
from reliable_proxy.certificate import OptimalityCertificate, certify_candidate
from reliable_proxy.domain import DispatchInstance
from reliable_proxy.oracle import ExactSolution, solve_exact
from reliable_proxy.projection import project_feasible
from reliable_proxy.proxy import ReliableOptimizationProxy


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    instance: str
    regime: str
    method: str
    objective: float
    exact_objective: float
    actual_gap: float | None
    actual_gap_percent: float | None
    certified_gap: float | None
    certified_gap_percent: float | None
    feasible: bool
    balance_violation: float
    bound_violation: float
    stationarity_residual: float | None
    accepted_proxy: bool
    used_fallback: bool
    inference_seconds: float
    repair_seconds: float
    certificate_seconds: float
    oracle_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    rows: tuple[BenchmarkRow, ...]
    summary: dict[str, dict[str, float | None]]
    certificate_curve: tuple[dict[str, float], ...]
    relative_gap_tolerance: float
    absolute_gap_tolerance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "summary": self.summary,
            "certificate_curve": list(self.certificate_curve),
            "relative_gap_tolerance": self.relative_gap_tolerance,
            "absolute_gap_tolerance": self.absolute_gap_tolerance,
        }


def _gap(value: float, reference: float) -> tuple[float, float]:
    absolute = max(0.0, value - reference)
    return absolute, 100.0 * absolute / max(1.0, abs(reference))



def _candidate_row(
    instance: DispatchInstance,
    exact: ExactSolution,
    *,
    method: str,
    decision: tuple[float, ...],
    certificate: OptimalityCertificate | None,
    accepted_proxy: bool = False,
    used_fallback: bool = False,
    inference_seconds: float = 0.0,
    repair_seconds: float = 0.0,
    certificate_seconds: float = 0.0,
    oracle_seconds: float = 0.0,
    total_seconds: float = 0.0,
) -> BenchmarkRow:
    audit = audit_feasibility(instance, decision)
    objective = instance.objective(decision)
    actual_gap: float | None
    actual_gap_percent: float | None
    if audit.feasible:
        actual_gap, actual_gap_percent = _gap(objective, exact.objective)
    else:
        actual_gap = None
        actual_gap_percent = None
    if certificate is not None and audit.feasible and actual_gap is not None:
        tolerance = 1e-7 * max(1.0, abs(objective), abs(exact.objective))
        if actual_gap > certificate.certified_gap + tolerance:
            raise RuntimeError(
                "dual certificate failed its oracle audit: "
                f"actual={actual_gap:.6g}, certified={certificate.certified_gap:.6g}"
            )
    bound_violation = max(audit.lower_bound_violation, audit.upper_bound_violation)
    return BenchmarkRow(
        instance=instance.name,
        regime=instance.regime,
        method=method,
        objective=objective,
        exact_objective=exact.objective,
        actual_gap=actual_gap,
        actual_gap_percent=actual_gap_percent,
        certified_gap=certificate.certified_gap if certificate is not None else None,
        certified_gap_percent=(
            certificate.certified_gap_percent if certificate is not None else None
        ),
        feasible=audit.feasible,
        balance_violation=audit.balance_violation,
        bound_violation=bound_violation,
        stationarity_residual=(
            certificate.kkt_audit.stationarity_residual if certificate is not None else None
        ),
        accepted_proxy=accepted_proxy,
        used_fallback=used_fallback,
        inference_seconds=inference_seconds,
        repair_seconds=repair_seconds,
        certificate_seconds=certificate_seconds,
        oracle_seconds=oracle_seconds,
        total_seconds=total_seconds,
    )


def _summarize(rows: list[BenchmarkRow]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for method in sorted({row.method for row in rows}):
        selected = [row for row in rows if row.method == method]
        gaps = [row.actual_gap_percent for row in selected if row.actual_gap_percent is not None]
        certificate_gaps = [
            row.certified_gap_percent
            for row in selected
            if row.certified_gap_percent is not None
        ]
        times = [row.total_seconds for row in selected]
        sorted_times = sorted(times)
        p95_index = min(len(sorted_times) - 1, math.ceil(0.95 * len(sorted_times)) - 1)
        summary[method] = {
            "instances": float(len(selected)),
            "feasibility_rate": statistics.fmean(float(row.feasible) for row in selected),
            "acceptance_rate": statistics.fmean(float(row.accepted_proxy) for row in selected),
            "fallback_rate": statistics.fmean(float(row.used_fallback) for row in selected),
            "mean_actual_gap_percent": statistics.fmean(gaps) if gaps else None,
            "max_actual_gap_percent": max(gaps) if gaps else None,
            "mean_certified_gap_percent": (
                statistics.fmean(certificate_gaps) if certificate_gaps else None
            ),
            "mean_total_seconds": statistics.fmean(times),
            "p95_total_seconds": sorted_times[p95_index],
        }
    return summary


def run_benchmark(
    instances: list[DispatchInstance] | tuple[DispatchInstance, ...],
    proxy: ReliableOptimizationProxy,
    *,
    relative_gap_tolerance: float = 0.01,
    absolute_gap_tolerance: float = 1e-6,
    curve_tolerances: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
) -> BenchmarkReport:
    if not instances:
        raise ValueError("instances must be nonempty")
    if relative_gap_tolerance < 0.0 or absolute_gap_tolerance < 0.0:
        raise ValueError("gap tolerances must be nonnegative")
    if any(tolerance < 0.0 for tolerance in curve_tolerances):
        raise ValueError("curve tolerances must be nonnegative")

    rows: list[BenchmarkRow] = []
    projected_records: list[tuple[float, float, float]] = []
    for instance in instances:
        exact = solve_exact(instance)
        raw, inference_seconds = proxy.predict_raw(instance)
        raw_started = time.perf_counter()
        raw_row = _candidate_row(
            instance,
            exact,
            method="raw_neural",
            decision=raw,
            certificate=None,
            inference_seconds=inference_seconds,
            total_seconds=inference_seconds + (time.perf_counter() - raw_started),
        )
        rows.append(raw_row)

        projection = project_feasible(instance, raw)
        certificate = certify_candidate(instance, projection.decision)
        projected_gap, _ = _gap(certificate.primal_objective, exact.objective)
        projected_records.append(
            (certificate.certified_gap, projected_gap, certificate.primal_objective)
        )
        rows.append(
            _candidate_row(
                instance,
                exact,
                method="projected_neural",
                decision=projection.decision,
                certificate=certificate,
                inference_seconds=inference_seconds,
                repair_seconds=projection.runtime_seconds,
                certificate_seconds=certificate.runtime_seconds,
                total_seconds=(
                    inference_seconds + projection.runtime_seconds + certificate.runtime_seconds
                ),
            )
        )
        for method, candidate in (
            ("proportional_baseline", proportional_dispatch(instance)),
            ("merit_order_baseline", merit_order_dispatch(instance)),
        ):
            started = time.perf_counter()
            baseline_certificate = certify_candidate(instance, candidate)
            rows.append(
                _candidate_row(
                    instance,
                    exact,
                    method=method,
                    decision=candidate,
                    certificate=baseline_certificate,
                    certificate_seconds=baseline_certificate.runtime_seconds,
                    total_seconds=time.perf_counter() - started,
                )
            )

        allowed_gap = absolute_gap_tolerance + relative_gap_tolerance * max(
            1.0, abs(certificate.primal_objective)
        )
        accepted = certificate.valid and certificate.certified_gap <= allowed_gap
        hybrid_decision = projection.decision if accepted else exact.decision
        hybrid_objective = instance.objective(hybrid_decision)
        hybrid_gap, _ = _gap(hybrid_objective, exact.objective)
        if accepted and hybrid_gap > allowed_gap + 1e-7 * max(1.0, abs(exact.objective)):
            raise RuntimeError("accepted proxy violated its configured quality envelope")
        rows.append(
            _candidate_row(
                instance,
                exact,
                method="certified_hybrid",
                decision=hybrid_decision,
                certificate=certificate if accepted else None,
                accepted_proxy=accepted,
                used_fallback=not accepted,
                inference_seconds=inference_seconds,
                repair_seconds=projection.runtime_seconds,
                certificate_seconds=certificate.runtime_seconds,
                oracle_seconds=0.0 if accepted else exact.runtime_seconds,
                total_seconds=(
                    inference_seconds
                    + projection.runtime_seconds
                    + certificate.runtime_seconds
                    + (0.0 if accepted else exact.runtime_seconds)
                ),
            )
        )
        rows.append(
            _candidate_row(
                instance,
                exact,
                method="exact_oracle",
                decision=exact.decision,
                certificate=certify_candidate(
                    instance,
                    exact.decision,
                    multiplier=exact.dual_multiplier,
                ),
                oracle_seconds=exact.runtime_seconds,
                total_seconds=exact.runtime_seconds,
            )
        )

    curve: list[dict[str, float]] = []
    for tolerance in curve_tolerances:
        accepted = [
            (certified_gap, actual_gap, objective)
            for certified_gap, actual_gap, objective in projected_records
            if certified_gap <= absolute_gap_tolerance + tolerance * max(1.0, abs(objective))
        ]
        violations = sum(
            actual_gap > certified_gap + 1e-7 * max(1.0, abs(objective))
            for certified_gap, actual_gap, objective in accepted
        )
        curve.append(
            {
                "relative_tolerance": tolerance,
                "acceptance_rate": len(accepted) / len(projected_records),
                "fallback_rate": 1.0 - len(accepted) / len(projected_records),
                "mean_actual_gap_percent": (
                    statistics.fmean(
                        100.0 * actual / max(1.0, abs(objective))
                        for _, actual, objective in accepted
                    )
                    if accepted
                    else 0.0
                ),
                "max_actual_gap_percent": (
                    max(
                        100.0 * actual / max(1.0, abs(objective))
                        for _, actual, objective in accepted
                    )
                    if accepted
                    else 0.0
                ),
                "guarantee_violations": float(violations),
            }
        )
    return BenchmarkReport(
        rows=tuple(rows),
        summary=_summarize(rows),
        certificate_curve=tuple(curve),
        relative_gap_tolerance=relative_gap_tolerance,
        absolute_gap_tolerance=absolute_gap_tolerance,
    )


def save_report_json(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_report_csv(report: BenchmarkReport, path: str | Path) -> None:
    rows = [row.to_dict() for row in report.rows]
    if not rows:
        raise ValueError("report has no rows")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
