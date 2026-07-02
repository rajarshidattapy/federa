"""Training metrics tracking across federated rounds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RoundMetrics:
    """Metrics reported by a single client for a single round."""

    round_number: int
    client_id: str
    num_samples: int
    loss: float
    accuracy: float | None = None
    duration_seconds: float | None = None


@dataclass(slots=True)
class MetricsTracker:
    """Aggregates per-client metrics into per-round summaries.

    The coordinator uses this to compute a sample-weighted average loss and
    accuracy across all clients that participated in a round, mirroring how
    FedAvg weights model updates by local dataset size.
    """

    history: list[RoundMetrics] = field(default_factory=list)

    def record(self, metrics: RoundMetrics) -> None:
        self.history.append(metrics)

    def round_summary(self, round_number: int) -> dict[str, float]:
        round_metrics = [m for m in self.history if m.round_number == round_number]
        if not round_metrics:
            return {}

        total_samples = sum(m.num_samples for m in round_metrics)
        if total_samples == 0:
            return {"num_clients": len(round_metrics)}

        weighted_loss = sum(m.loss * m.num_samples for m in round_metrics) / total_samples

        accuracies = [(m.accuracy, m.num_samples) for m in round_metrics if m.accuracy is not None]
        weighted_accuracy = None
        if accuracies:
            acc_samples = sum(n for _, n in accuracies)
            weighted_accuracy = sum(a * n for a, n in accuracies) / acc_samples

        summary = {
            "round": float(round_number),
            "num_clients": float(len(round_metrics)),
            "total_samples": float(total_samples),
            "loss": weighted_loss,
        }
        if weighted_accuracy is not None:
            summary["accuracy"] = weighted_accuracy
        return summary

    def as_dict(self) -> dict[int, dict[str, float]]:
        rounds = sorted({m.round_number for m in self.history})
        return {r: self.round_summary(r) for r in rounds}
