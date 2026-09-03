from __future__ import annotations

import pytest

from reliable_proxy.domain import DispatchInstance


@pytest.fixture
def hand_instance() -> DispatchInstance:
    return DispatchInstance(
        name="hand",
        quadratic_costs=(1.0, 2.0, 3.0),
        linear_costs=(1.0, 2.0, 3.0),
        lower_bounds=(0.0, 0.0, 0.0),
        upper_bounds=(10.0, 10.0, 10.0),
        demand=6.0,
    )


@pytest.fixture
def boundary_instance() -> DispatchInstance:
    return DispatchInstance(
        name="boundary",
        quadratic_costs=(1.0, 2.0),
        linear_costs=(1.0, 1.0),
        lower_bounds=(1.0, 2.0),
        upper_bounds=(4.0, 5.0),
        demand=3.0,
    )
