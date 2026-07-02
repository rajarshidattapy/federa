"""Local training on a client's private data.

Successor to `SwarmNode.trainLocalBatchAsync` in the TypeScript prototype's
client-node.ts, but delegates gradient computation to PyTorch autograd/optim
instead of hand-rolled backprop, works with any loss function, and
optionally applies the FedProx proximal term (`federa.training.fedprox`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from federa.models.base import FederatedModel
from federa.privacy.clipping import clip_parameter_gradients_
from federa.training.fedprox import fedprox_proximal_term
from federa.training.optimizer import OptimizerName, build_optimizer
from federa.utils.config import ClientSettings

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass(slots=True)
class LocalTrainingResult:
    weights: dict[str, torch.Tensor]
    num_samples: int
    loss: float
    metric: float | None
    duration_seconds: float


class LocalTrainer:
    def __init__(
        self,
        model: FederatedModel,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module | None = None,
        metric_fn: MetricFn | None = None,
        settings: ClientSettings | None = None,
        optimizer_name: OptimizerName = "sgd",
        fedprox_mu: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.metric_fn = metric_fn
        self.fedprox_mu = fedprox_mu
        self.optimizer = build_optimizer(
            model.parameters(),
            name=optimizer_name,
            lr=settings.learning_rate if settings else 0.01,
        )

    def train_round(
        self, epochs: int = 1, max_grad_norm: float | None = None
    ) -> LocalTrainingResult:
        self.model.train_mode()
        global_weights = (
            [p.detach().clone() for p in self.model.parameters()] if self.fedprox_mu > 0 else []
        )

        start = time.monotonic()
        total_loss = 0.0
        total_metric = 0.0
        num_samples = 0

        for epoch in range(epochs):
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model.forward(inputs)
                loss = self.loss_fn(predictions, targets)
                if self.fedprox_mu > 0:
                    loss = loss + fedprox_proximal_term(
                        self.model.parameters(), global_weights, self.fedprox_mu
                    )
                loss.backward()

                if max_grad_norm is not None:
                    clip_parameter_gradients_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()

                batch_size = inputs.shape[0]
                total_loss += float(loss.item()) * batch_size
                if self.metric_fn is not None:
                    total_metric += self.metric_fn(predictions.detach(), targets) * batch_size
                if epoch == 0:
                    num_samples += batch_size

        duration = time.monotonic() - start
        divisor = num_samples * epochs
        avg_loss = total_loss / divisor if divisor else 0.0
        avg_metric = (total_metric / divisor) if divisor and self.metric_fn is not None else None

        return LocalTrainingResult(
            weights=self.model.get_weights(),
            num_samples=num_samples,
            loss=avg_loss,
            metric=avg_metric,
            duration_seconds=duration,
        )
