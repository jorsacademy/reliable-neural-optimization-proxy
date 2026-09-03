"""Oracle-labelled datasets for the optimization proxy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from reliable_proxy.domain import DispatchInstance
from reliable_proxy.features import FEATURE_SCHEMA_VERSION, decision_to_canonical, featurize, feature_names
from reliable_proxy.oracle import solve_exact


@dataclass(frozen=True, slots=True)
class ProxyDataset:
    features: np.ndarray
    targets: np.ndarray
    instance_names: tuple[str, ...]
    regimes: tuple[str, ...]
    feature_names: tuple[str, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("features and targets must be matrices")
        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("features and targets must have the same row count")
        if len(self.instance_names) != self.features.shape[0]:
            raise ValueError("instance_names have the wrong length")
        if len(self.regimes) != self.features.shape[0]:
            raise ValueError("regimes have the wrong length")
        if len(self.feature_names) != self.features.shape[1]:
            raise ValueError("feature_names have the wrong length")
        if not np.all(np.isfinite(self.features)) or not np.all(np.isfinite(self.targets)):
            raise ValueError("dataset arrays must be finite")

    @property
    def sample_count(self) -> int:
        return self.features.shape[0]

    @property
    def unit_count(self) -> int:
        return self.targets.shape[1]

    def summary(self) -> dict[str, object]:
        return {
            "samples": self.sample_count,
            "unit_count": self.unit_count,
            "input_features": self.features.shape[1],
            "regimes": sorted(set(self.regimes)),
            "feature_schema_version": self.metadata.get("feature_schema_version"),
        }


def build_dataset(instances: list[DispatchInstance] | tuple[DispatchInstance, ...]) -> ProxyDataset:
    if not instances:
        raise ValueError("instances must be nonempty")
    unit_count = instances[0].unit_count
    if any(instance.unit_count != unit_count for instance in instances):
        raise ValueError("all instances must use the same unit_count")
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    names: list[str] = []
    regimes: list[str] = []
    oracle_iterations: list[int] = []
    for instance in instances:
        canonical = featurize(instance)
        optimum = solve_exact(instance)
        feature_rows.append(canonical.values)
        target_rows.append(decision_to_canonical(optimum.decision, canonical.order))
        names.append(instance.name)
        regimes.append(instance.regime)
        oracle_iterations.append(optimum.iterations)
    return ProxyDataset(
        features=np.vstack(feature_rows),
        targets=np.vstack(target_rows),
        instance_names=tuple(names),
        regimes=tuple(regimes),
        feature_names=feature_names(unit_count),
        metadata={
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "unit_count": unit_count,
            "oracle": "strictly-convex KKT bisection",
            "oracle_iteration_mean": float(np.mean(oracle_iterations)),
        },
    )


def save_dataset(dataset: ProxyDataset, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=dataset.features,
        targets=dataset.targets,
        instance_names=np.asarray(dataset.instance_names),
        regimes=np.asarray(dataset.regimes),
        feature_names=np.asarray(dataset.feature_names),
        metadata_json=np.asarray([json.dumps(dataset.metadata, sort_keys=True)]),
    )


def load_dataset(path: str | Path) -> ProxyDataset:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = cast(
            dict[str, object],
            json.loads(str(payload["metadata_json"][0])),
        )
        return ProxyDataset(
            features=np.asarray(payload["features"], dtype=float),
            targets=np.asarray(payload["targets"], dtype=float),
            instance_names=tuple(str(value) for value in payload["instance_names"]),
            regimes=tuple(str(value) for value in payload["regimes"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            metadata=metadata,
        )
