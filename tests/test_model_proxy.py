from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reliable_proxy.dataset import build_dataset
from reliable_proxy.features import FEATURE_SCHEMA_VERSION, featurize
from reliable_proxy.generator import GeneratorConfig, generate_instance, generate_instances
from reliable_proxy.model import NumpyMLPProxy, TrainingConfig
from reliable_proxy.oracle import solve_exact
from reliable_proxy.proxy import ReliableOptimizationProxy
from reliable_proxy.training import train_proxy


def _small_trained_proxy() -> ReliableOptimizationProxy:
    train = build_dataset(
        generate_instances(120, unit_count=6, regime="in_distribution", seed=1000)
    )
    validation = build_dataset(
        generate_instances(30, unit_count=6, regime="in_distribution", seed=3000)
    )
    proxy, _ = train_proxy(
        train,
        validation,
        config=TrainingConfig(
            hidden_dim=20,
            epochs=12,
            batch_size=32,
            patience=6,
            seed=7,
        ),
    )
    return proxy


def test_model_training_reduces_loss_and_checkpoint_round_trip(tmp_path: Path) -> None:
    train = build_dataset(
        generate_instances(100, unit_count=5, regime="in_distribution", seed=10)
    )
    validation = build_dataset(
        generate_instances(24, unit_count=5, regime="in_distribution", seed=500)
    )
    proxy, report = train_proxy(
        train,
        validation,
        config=TrainingConfig(
            hidden_dim=16,
            epochs=10,
            batch_size=25,
            patience=5,
            seed=3,
        ),
    )
    assert report.history.train_losses[-1] < report.history.train_losses[0]
    path = tmp_path / "proxy.npz"
    proxy.model.save(path)
    loaded = ReliableOptimizationProxy.load(path)
    sample = generate_instance(GeneratorConfig(unit_count=5, seed=900))
    assert loaded.predict_raw(sample)[0] == pytest.approx(proxy.predict_raw(sample)[0])


def test_perfect_constant_model_is_accepted() -> None:
    instance = generate_instance(GeneratorConfig(unit_count=4, seed=44))
    canonical = featurize(instance)
    optimum = solve_exact(instance)
    target = np.asarray(optimum.decision)[np.asarray(canonical.order)]
    model = NumpyMLPProxy(canonical.values.size, instance.unit_count, hidden_dim=4, seed=0)
    model.w1.fill(0.0)
    model.w2.fill(0.0)
    model.w3.fill(0.0)
    model.b1.fill(0.0)
    model.b2.fill(0.0)
    model.b3.fill(0.0)
    model.output_mean = target.copy()
    model.output_scale = np.ones_like(target)
    model.metadata.update(
        {"unit_count": instance.unit_count, "feature_schema_version": FEATURE_SCHEMA_VERSION}
    )
    proxy = ReliableOptimizationProxy(model, unit_count=instance.unit_count)
    result = proxy.solve(instance, relative_gap_tolerance=1e-8, absolute_gap_tolerance=1e-6)
    assert result.accepted_proxy
    assert not result.used_fallback
    assert result.final_objective == pytest.approx(optimum.objective, abs=1e-6)


def test_untrained_model_falls_back_under_zero_gap_tolerance() -> None:
    instance = generate_instance(GeneratorConfig(unit_count=5, regime="high_demand", seed=77))
    feature_count = featurize(instance).values.size
    model = NumpyMLPProxy(feature_count, instance.unit_count, hidden_dim=5, seed=1)
    model.metadata.update(
        {"unit_count": instance.unit_count, "feature_schema_version": FEATURE_SCHEMA_VERSION}
    )
    proxy = ReliableOptimizationProxy(model, unit_count=instance.unit_count)
    result = proxy.solve(instance, relative_gap_tolerance=0.0, absolute_gap_tolerance=0.0)
    assert result.used_fallback
    assert result.exact_solution is not None
    assert result.final_objective == pytest.approx(result.exact_solution.objective)


def test_trained_proxy_returns_feasible_decision() -> None:
    proxy = _small_trained_proxy()
    instance = generate_instance(
        GeneratorConfig(unit_count=6, regime="capacity_shift", seed=8000)
    )
    result = proxy.solve(instance, relative_gap_tolerance=0.05)
    assert result.projection.audit.feasible
    assert result.certificate.valid
