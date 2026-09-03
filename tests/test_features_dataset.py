from __future__ import annotations

from pathlib import Path

import numpy as np

from reliable_proxy.dataset import build_dataset, load_dataset, save_dataset
from reliable_proxy.domain import DispatchInstance
from reliable_proxy.features import decision_from_canonical, decision_to_canonical, featurize
from reliable_proxy.generator import generate_instances


def _permuted(instance: DispatchInstance, permutation: tuple[int, ...]) -> DispatchInstance:
    return DispatchInstance(
        name=f"{instance.name}-permuted",
        regime=instance.regime,
        quadratic_costs=tuple(instance.quadratic_costs[index] for index in permutation),
        linear_costs=tuple(instance.linear_costs[index] for index in permutation),
        lower_bounds=tuple(instance.lower_bounds[index] for index in permutation),
        upper_bounds=tuple(instance.upper_bounds[index] for index in permutation),
        demand=instance.demand,
    )


def test_canonical_features_are_permutation_stable(hand_instance: DispatchInstance) -> None:
    permutation = (2, 0, 1)
    permuted = _permuted(hand_instance, permutation)
    original_features = featurize(hand_instance)
    permuted_features = featurize(permuted)
    assert np.allclose(original_features.values, permuted_features.values)

    decision = np.asarray((3.0, 2.0, 1.0))
    canonical = decision_to_canonical(decision, original_features.order)
    restored = decision_from_canonical(canonical, original_features.order)
    assert np.array_equal(restored, decision)


def test_dataset_round_trip_and_summary(tmp_path: Path) -> None:
    dataset = build_dataset(
        generate_instances(8, unit_count=5, regime="in_distribution", seed=100)
    )
    path = tmp_path / "dataset.npz"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert np.array_equal(loaded.features, dataset.features)
    assert np.array_equal(loaded.targets, dataset.targets)
    assert loaded.feature_names == dataset.feature_names
    assert loaded.summary()["samples"] == 8
