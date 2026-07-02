"""Model adapters that let Federa drive arbitrary PyTorch models."""

from federa.models.base import FederatedModel
from federa.models.pytorch import TorchModelAdapter, wrap_model

__all__ = ["FederatedModel", "TorchModelAdapter", "wrap_model"]
