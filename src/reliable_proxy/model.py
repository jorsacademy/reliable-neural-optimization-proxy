"""A compact auditable NumPy neural regressor for dispatch solutions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

CHECKPOINT_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    hidden_dim: int = 64
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    patience: int = 15
    gradient_clip: float = 10.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("hidden_dim, epochs, and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.patience <= 0 or self.gradient_clip <= 0.0:
            raise ValueError("patience and gradient_clip must be positive")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    stopped_early: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "train_losses": list(self.train_losses),
            "validation_losses": list(self.validation_losses),
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
        }


class NumpyMLPProxy:
    """Two-hidden-layer tanh MLP trained with Adam and early stopping."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        *,
        seed: int = 0,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_mean = np.zeros(input_dim, dtype=float)
        self.input_scale = np.ones(input_dim, dtype=float)
        self.output_mean = np.zeros(output_dim, dtype=float)
        self.output_scale = np.ones(output_dim, dtype=float)
        self.w1 = rng.normal(0.0, np.sqrt(1.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim, dtype=float)
        self.w3 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, output_dim))
        self.b3 = np.zeros(output_dim, dtype=float)
        self.metadata: dict[str, object] = {}

    def _standardize_inputs(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape (n, {self.input_dim})")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return (values - self.input_mean) / self.input_scale

    def _standardize_targets(self, targets: np.ndarray) -> np.ndarray:
        values = np.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError(f"targets must have shape (n, {self.output_dim})")
        if not np.all(np.isfinite(values)):
            raise ValueError("targets must be finite")
        return (values - self.output_mean) / self.output_scale

    def _forward(self, standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden1 = np.tanh(standardized @ self.w1 + self.b1)
        hidden2 = np.tanh(hidden1 @ self.w2 + self.b2)
        output = hidden2 @ self.w3 + self.b3
        return hidden1, hidden2, output

    def predict(self, features: np.ndarray) -> np.ndarray:
        _, _, output = self._forward(self._standardize_inputs(features))
        return output * self.output_scale + self.output_mean

    def _loss(self, x: np.ndarray, y: np.ndarray, weight_decay: float) -> float:
        prediction = self._forward(x)[2]
        mse = float(np.mean((prediction - y) ** 2))
        penalty = 0.5 * weight_decay * float(
            np.sum(self.w1 * self.w1)
            + np.sum(self.w2 * self.w2)
            + np.sum(self.w3 * self.w3)
        )
        return mse + penalty

    def fit(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        config: TrainingConfig | None = None,
    ) -> TrainingHistory:
        config = config or TrainingConfig(hidden_dim=self.hidden_dim)
        if config.hidden_dim != self.hidden_dim:
            raise ValueError("config.hidden_dim must match the model hidden_dim")
        train_x = np.asarray(train_features, dtype=float)
        train_y = np.asarray(train_targets, dtype=float)
        validation_x = np.asarray(validation_features, dtype=float)
        validation_y = np.asarray(validation_targets, dtype=float)
        if train_x.shape[0] == 0 or validation_x.shape[0] == 0:
            raise ValueError("training and validation sets must be nonempty")
        if train_x.ndim != 2 or train_x.shape[1] != self.input_dim:
            raise ValueError("train_features have the wrong shape")
        if validation_x.ndim != 2 or validation_x.shape[1] != self.input_dim:
            raise ValueError("validation_features have the wrong shape")
        if train_y.shape != (train_x.shape[0], self.output_dim):
            raise ValueError("train_targets have the wrong shape")
        if validation_y.shape != (validation_x.shape[0], self.output_dim):
            raise ValueError("validation_targets have the wrong shape")
        if not all(
            np.all(np.isfinite(values))
            for values in (train_x, train_y, validation_x, validation_y)
        ):
            raise ValueError("training arrays must be finite")

        self.input_mean = np.mean(train_x, axis=0)
        input_scale = np.std(train_x, axis=0)
        self.input_scale = np.where(input_scale > 1e-10, input_scale, 1.0)
        self.output_mean = np.mean(train_y, axis=0)
        output_scale = np.std(train_y, axis=0)
        self.output_scale = np.where(output_scale > 1e-10, output_scale, 1.0)
        x_train = self._standardize_inputs(train_x)
        y_train = self._standardize_targets(train_y)
        x_validation = self._standardize_inputs(validation_x)
        y_validation = self._standardize_targets(validation_y)

        parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        step = 0
        rng = np.random.default_rng(config.seed)
        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_validation = float("inf")
        best_epoch = 0
        best_parameters = [parameter.copy() for parameter in parameters]
        stale_epochs = 0

        for epoch in range(config.epochs):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, len(order), config.batch_size):
                indices = order[start : start + config.batch_size]
                xb = x_train[indices]
                yb = y_train[indices]
                hidden1, hidden2, prediction = self._forward(xb)
                batch_size = max(1, len(indices))
                grad_output = 2.0 * (prediction - yb) / (batch_size * self.output_dim)
                grad_w3 = hidden2.T @ grad_output + config.weight_decay * self.w3
                grad_b3 = np.sum(grad_output, axis=0)
                grad_hidden2 = grad_output @ self.w3.T
                grad_pre2 = grad_hidden2 * (1.0 - hidden2 * hidden2)
                grad_w2 = hidden1.T @ grad_pre2 + config.weight_decay * self.w2
                grad_b2 = np.sum(grad_pre2, axis=0)
                grad_hidden1 = grad_pre2 @ self.w2.T
                grad_pre1 = grad_hidden1 * (1.0 - hidden1 * hidden1)
                grad_w1 = xb.T @ grad_pre1 + config.weight_decay * self.w1
                grad_b1 = np.sum(grad_pre1, axis=0)
                gradients = [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]
                norm = float(np.sqrt(sum(np.sum(gradient * gradient) for gradient in gradients)))
                if norm > config.gradient_clip:
                    gradients = [
                        gradient * config.gradient_clip / max(norm, 1e-12)
                        for gradient in gradients
                    ]
                step += 1
                for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients, strict=True)
                ):
                    first_moments[index] = beta1 * first_moments[index] + (1.0 - beta1) * gradient
                    second_moments[index] = (
                        beta2 * second_moments[index] + (1.0 - beta2) * gradient * gradient
                    )
                    first_hat = first_moments[index] / (1.0 - beta1**step)
                    second_hat = second_moments[index] / (1.0 - beta2**step)
                    parameter -= config.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
            train_loss = self._loss(x_train, y_train, config.weight_decay)
            validation_loss = self._loss(x_validation, y_validation, 0.0)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation - 1e-10:
                best_validation = validation_loss
                best_epoch = epoch
                best_parameters = [parameter.copy() for parameter in parameters]
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= config.patience:
                break

        for parameter, best in zip(parameters, best_parameters, strict=True):
            parameter[...] = best
        return TrainingHistory(
            train_losses=tuple(train_losses),
            validation_losses=tuple(validation_losses),
            best_epoch=best_epoch,
            stopped_early=len(train_losses) < config.epochs,
        )

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined = dict(self.metadata)
        if metadata:
            combined.update(metadata)
        combined.update(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
            }
        )
        np.savez_compressed(
            output,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            output_mean=self.output_mean,
            output_scale=self.output_scale,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            metadata_json=np.asarray([json.dumps(combined, sort_keys=True)]),
        )

    @classmethod
    def load(cls, path: str | Path) -> NumpyMLPProxy:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = cast(
                dict[str, object],
                json.loads(str(payload["metadata_json"][0])),
            )
            if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
                raise ValueError("unsupported checkpoint version")
            input_dim = int(cast(int, metadata["input_dim"]))
            output_dim = int(cast(int, metadata["output_dim"]))
            hidden_dim = int(cast(int, metadata["hidden_dim"]))
            model = cls(input_dim, output_dim, hidden_dim, seed=0)
            for name in (
                "input_mean",
                "input_scale",
                "output_mean",
                "output_scale",
                "w1",
                "b1",
                "w2",
                "b2",
                "w3",
                "b3",
            ):
                setattr(model, name, np.asarray(payload[name], dtype=float))
            model.metadata = metadata
        return model
