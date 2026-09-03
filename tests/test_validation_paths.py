from __future__ import annotations

import numpy as np
import pytest

from reliable_proxy.certificate import estimate_multiplier
from reliable_proxy.dataset import build_dataset
from reliable_proxy.domain import DispatchInstance
from reliable_proxy.features import decision_from_canonical, feature_names
from reliable_proxy.generator import GeneratorConfig, generate_instance
from reliable_proxy.model import NumpyMLPProxy, TrainingConfig
from reliable_proxy.oracle import solve_exact
from reliable_proxy.projection import project_feasible
from reliable_proxy.proxy import ReliableOptimizationProxy


def test_domain_and_generator_validation_paths() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        DispatchInstance("", (1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (2.0, 2.0), 2.0)
    with pytest.raises(ValueError, match="same length"):
        DispatchInstance("bad", (1.0, 1.0), (1.0,), (0.0, 0.0), (2.0, 2.0), 2.0)
    with pytest.raises(ValueError, match="at least two"):
        GeneratorConfig(unit_count=1)
    with pytest.raises(ValueError, match="either config"):
        generate_instance(GeneratorConfig(), seed=2)
    with pytest.raises(ValueError, match="unknown regime"):
        DispatchInstance.from_dict(
            {
                "name": "bad",
                "regime": "unknown",
                "quadratic_costs": [1.0, 1.0],
                "linear_costs": [1.0, 1.0],
                "lower_bounds": [0.0, 0.0],
                "upper_bounds": [2.0, 2.0],
                "demand": 2.0,
            }
        )


def test_upper_boundary_oracle_projection_and_multiplier() -> None:
    instance = DispatchInstance(
        "upper",
        (1.0, 2.0),
        (1.0, 2.0),
        (0.0, 0.0),
        (3.0, 4.0),
        7.0,
    )
    exact = solve_exact(instance)
    projected = project_feasible(instance, (-10.0, -20.0))
    assert exact.decision == instance.upper_bounds
    assert projected.decision == pytest.approx(instance.upper_bounds)
    assert np.isfinite(estimate_multiplier(instance, projected.decision))


def test_shape_and_configuration_guards(hand_instance: DispatchInstance) -> None:
    with pytest.raises(ValueError, match="unit_count"):
        feature_names(1)
    with pytest.raises(ValueError, match="incompatible"):
        decision_from_canonical((1.0, 2.0), (0, 1, 2))
    with pytest.raises(ValueError, match="wrong shape"):
        project_feasible(hand_instance, (1.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="same unit_count"):
        build_dataset(
            [
                generate_instance(GeneratorConfig(unit_count=3, seed=1)),
                generate_instance(GeneratorConfig(unit_count=4, seed=2)),
            ]
        )
    model = NumpyMLPProxy(len(feature_names(3)), 3, hidden_dim=4)
    proxy = ReliableOptimizationProxy(model, unit_count=3)
    with pytest.raises(ValueError, match="expects 3"):
        proxy.predict_raw(generate_instance(GeneratorConfig(unit_count=4, seed=3)))
