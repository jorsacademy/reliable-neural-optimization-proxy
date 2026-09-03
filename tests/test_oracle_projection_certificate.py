from __future__ import annotations

import numpy as np
import pytest

from reliable_proxy.audit import audit_feasibility
from reliable_proxy.baselines import merit_order_dispatch, proportional_dispatch
from reliable_proxy.certificate import certify_candidate
from reliable_proxy.domain import DispatchInstance
from reliable_proxy.generator import generate_instances
from reliable_proxy.oracle import dual_value, solve_exact
from reliable_proxy.projection import project_feasible


def test_exact_oracle_satisfies_strong_duality_and_known_solution(
    hand_instance: DispatchInstance,
) -> None:
    result = solve_exact(hand_instance)
    assert result.decision == pytest.approx((3.9090909091, 1.4545454545, 0.6363636364), abs=1e-8)
    assert result.duality_gap <= 1e-8
    assert result.stationarity_residual <= 1e-8
    assert dual_value(hand_instance, result.dual_multiplier) == pytest.approx(result.objective)


def test_oracle_and_projection_handle_aggregate_lower_boundary(
    boundary_instance: DispatchInstance,
) -> None:
    exact = solve_exact(boundary_instance)
    projection = project_feasible(boundary_instance, (100.0, -100.0))
    assert exact.decision == boundary_instance.lower_bounds
    assert projection.decision == pytest.approx(boundary_instance.lower_bounds)
    assert projection.audit.feasible


def test_projection_is_feasible_and_idempotent(hand_instance: DispatchInstance) -> None:
    raw = np.asarray((-5.0, 20.0, 1.0))
    projected = project_feasible(hand_instance, raw)
    assert audit_feasibility(hand_instance, projected.decision).feasible
    repeated = project_feasible(hand_instance, projected.decision)
    assert repeated.decision == pytest.approx(projected.decision, abs=1e-9)
    assert projected.correction_norm > 0.0


def test_dual_certificate_upper_bounds_actual_suboptimality(
    hand_instance: DispatchInstance,
) -> None:
    exact = solve_exact(hand_instance)
    for candidate in (proportional_dispatch(hand_instance), merit_order_dispatch(hand_instance)):
        certificate = certify_candidate(hand_instance, candidate)
        actual_gap = hand_instance.objective(candidate) - exact.objective
        assert certificate.valid
        assert 0.0 <= actual_gap <= certificate.certified_gap + 1e-8


def test_infeasible_candidate_does_not_receive_valid_certificate(
    hand_instance: DispatchInstance,
) -> None:
    certificate = certify_candidate(hand_instance, (0.0, 0.0, 0.0))
    assert not certificate.valid


def test_certificate_property_holds_on_generated_candidates() -> None:
    for instance in generate_instances(
        12,
        unit_count=6,
        regime="combined_shift",
        seed=3000,
    ):
        exact = solve_exact(instance)
        candidate = proportional_dispatch(instance)
        certificate = certify_candidate(instance, candidate)
        assert instance.objective(candidate) - exact.objective <= certificate.certified_gap + 1e-6
