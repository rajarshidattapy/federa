"""The central FedAvg server: aggregation, client routing, and global state."""

from federa.coordinator.aggregator import Aggregator
from federa.coordinator.routing import ClientRegistry, ConnectedClient
from federa.coordinator.server import Coordinator
from federa.coordinator.state import GlobalState

__all__ = ["Aggregator", "ClientRegistry", "ConnectedClient", "Coordinator", "GlobalState"]
