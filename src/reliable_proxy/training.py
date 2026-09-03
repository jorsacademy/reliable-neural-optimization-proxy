"""Training helpers and regression diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reliable_proxy.dataset import ProxyDataset
from reliable_proxy.features import FEATURE_SCHEMA_VERSION
from reliable_proxy.model import NumpyMLPProxy, TrainingConfig, TrainingHistory
from reliable_proxy.proxy import ReliableOptimizationProxy


@dataclass(frozen=True, slots=True)
class TrainingReport:
    history: TrainingHistory
    validation_rmse: float
    validation_mae: float
    validation_max_abs_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "history": self.history.to_dict(),
            "validation_rmse": self.validation_rmse,
            "validation_mae": self.validation_mae,
            "validation_max_abs_error": self.validation_max_abs_error,
        }


def train_proxy(
    train_dataset: ProxyDataset,
    validation_dataset: ProxyDataset,
    *,
    config: TrainingConfig | None = None,
) -> tuple[ReliableOptimizationProxy, TrainingReport]:
    if train_dataset.unit_count != validation_dataset.unit_count:
        raise ValueError("training and validation unit_count values differ")
    if train_dataset.feature_names != validation_dataset.feature_names:
        raise ValueError("training and validation feature schemas differ")
    config = config or TrainingConfig()
    model = NumpyMLPProxy(
        input_dim=train_dataset.features.shape[1],
        output_dim=train_dataset.unit_count,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )
    history = model.fit(
        train_dataset.features,
        train_dataset.targets,
        validation_dataset.features,
        validation_dataset.targets,
        config,
    )
    predictions = model.predict(validation_dataset.features)
    errors = predictions - validation_dataset.targets
    report = TrainingReport(
        history=history,
        validation_rmse=float(np.sqrt(np.mean(errors * errors))),
        validation_mae=float(np.mean(np.abs(errors))),
        validation_max_abs_error=float(np.max(np.abs(errors))),
    )
    model.metadata.update(
        {
            "unit_count": train_dataset.unit_count,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_samples": train_dataset.sample_count,
            "validation_samples": validation_dataset.sample_count,
            "validation_rmse": report.validation_rmse,
            "validation_mae": report.validation_mae,
            "training_config": {
                "hidden_dim": config.hidden_dim,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "patience": config.patience,
                "gradient_clip": config.gradient_clip,
                "seed": config.seed,
            },
        }
    )
    return ReliableOptimizationProxy(model, unit_count=train_dataset.unit_count), report
