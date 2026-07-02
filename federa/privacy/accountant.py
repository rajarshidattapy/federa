"""Privacy budget accounting across federated rounds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PrivacyAccountant:
    """Tracks cumulative privacy spend using basic sequential composition.

    Sequential composition (Dwork & Roth, Theorem 3.16) says that running k
    mechanisms with budgets (epsilon_i, delta_i) is at worst
    (sum(epsilon_i), sum(delta_i))-DP overall. This is a safe, conservative
    default; use `rdp_epsilon` for the much tighter bound Opacus computes
    via an RDP accountant when many rounds are involved.
    """

    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def spend(self, round_number: int, epsilon: float, delta: float = 0.0) -> None:
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative")
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.history.append((round_number, epsilon, delta))

    def remaining_budget(self, total_epsilon: float) -> float:
        return max(0.0, total_epsilon - self.spent_epsilon)

    def is_exhausted(self, total_epsilon: float) -> bool:
        return self.spent_epsilon >= total_epsilon


def rdp_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """Tight epsilon via Opacus' Renyi-DP accountant.

    Requires the optional `federa[privacy]` extra. Raises `ImportError` if
    Opacus is not installed -- callers should catch that and fall back to
    `PrivacyAccountant`'s conservative composition.
    """
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    epsilon, _ = accountant.get_privacy_spent(delta=delta)
    return float(epsilon)
