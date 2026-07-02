import pytest
import torch

from federa.training.fedavg import ClientUpdate, federated_average
from federa.training.fedprox import fedprox_proximal_term


def test_federated_average_weights_by_sample_count():
    updates = [
        ClientUpdate(client_id="a", num_samples=1, weights={"w": torch.tensor([0.0])}),
        ClientUpdate(client_id="b", num_samples=3, weights={"w": torch.tensor([4.0])}),
    ]
    result = federated_average(updates)
    # weighted mean = (1*0 + 3*4) / 4 = 3.0
    assert torch.allclose(result["w"], torch.tensor([3.0]))


def test_federated_average_rejects_empty_list():
    with pytest.raises(ValueError):
        federated_average([])


def test_federated_average_rejects_mismatched_keys():
    updates = [
        ClientUpdate(client_id="a", num_samples=1, weights={"w": torch.tensor([1.0])}),
        ClientUpdate(client_id="b", num_samples=1, weights={"v": torch.tensor([1.0])}),
    ]
    with pytest.raises(ValueError):
        federated_average(updates)


def test_federated_average_keeps_non_float_buffers_from_latest_client():
    updates = [
        ClientUpdate(
            client_id="a", num_samples=1, weights={"n": torch.tensor(1, dtype=torch.int64)}
        ),
        ClientUpdate(
            client_id="b", num_samples=1, weights={"n": torch.tensor(2, dtype=torch.int64)}
        ),
    ]
    result = federated_average(updates)
    assert int(result["n"]) == 2


def test_fedprox_proximal_term_zero_when_mu_zero():
    local = [torch.nn.Parameter(torch.randn(3))]
    global_weights = [torch.randn(3)]
    term = fedprox_proximal_term(local, global_weights, mu=0.0)
    assert float(term) == 0.0


def test_fedprox_proximal_term_matches_closed_form():
    local = [torch.nn.Parameter(torch.tensor([1.0, 1.0]))]
    global_weights = [torch.tensor([0.0, 0.0])]
    term = fedprox_proximal_term(local, global_weights, mu=1.0)
    assert float(term) == pytest.approx(1.0)  # (1/2) * (1^2 + 1^2) = 1.0
