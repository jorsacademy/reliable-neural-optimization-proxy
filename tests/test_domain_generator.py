from __future__ import annotations

from pathlib import Path

import pytest

from reliable_proxy.domain import DispatchInstance, load_instance, save_instance
from reliable_proxy.generator import GeneratorConfig, generate_instance


def test_instance_json_round_trip(hand_instance: DispatchInstance, tmp_path: Path) -> None:
    path = tmp_path / "instance.json"
    save_instance(hand_instance, path)
    assert load_instance(path) == hand_instance
    assert DispatchInstance.from_dict(hand_instance.to_dict()) == hand_instance


def test_instance_rejects_invalid_bounds_and_demand() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        DispatchInstance("bad", (0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0), 1.0)
    with pytest.raises(ValueError, match="aggregate"):
        DispatchInstance("bad", (1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0), 3.0)


def test_generator_is_deterministic_and_regimes_are_feasible() -> None:
    first = generate_instance(GeneratorConfig(unit_count=6, regime="high_demand", seed=17))
    second = generate_instance(GeneratorConfig(unit_count=6, regime="high_demand", seed=17))
    assert first == second
    assert 0.84 <= first.demand_ratio <= 0.97
    shifted = generate_instance(GeneratorConfig(unit_count=6, regime="cost_shift", seed=17))
    assert min(shifted.linear_costs) >= 7.0
