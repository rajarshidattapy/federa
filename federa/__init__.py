"""Federa: decentralized federated learning infrastructure for PyTorch.

Inspired by "Communication-Efficient Learning of Deep Networks from
Decentralized Data" (McMahan et al., 2017):
https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
"""

from federa.client.node import SwarmNode
from federa.coordinator.server import Coordinator

__version__ = "0.1.0"

__all__ = ["SwarmNode", "Coordinator", "__version__"]
