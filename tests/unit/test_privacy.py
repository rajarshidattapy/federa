import pytest
import torch
import torch.nn as nn

from federa.privacy.accountant import PrivacyAccountant
from federa.privacy.clipping import clip_parameter_gradients_, clip_state_dict_
from federa.privacy.gaussian import GaussianMechanism
from federa.privacy.laplace import LaplaceMechanism


def test_laplace_noise_changes_values_but_preserves_shape():
    mechanism = LaplaceMechanism(epsilon=0.5)
    tensor = torch.zeros(100)
    noised = mechanism.add_noise(tensor)
    assert noised.shape == tensor.shape
    assert not torch.allclose(noised, tensor)


def test_laplace_lower_epsilon_yields_larger_noise_scale():
    assert LaplaceMechanism(epsilon=0.1).scale > LaplaceMechanism(epsilon=10.0).scale


def test_laplace_rejects_nonpositive_epsilon():
    with pytest.raises(ValueError):
        LaplaceMechanism(epsilon=0)


def test_gaussian_sigma_is_positive_and_scales_with_delta_tightness():
    loose = GaussianMechanism(epsilon=1.0, delta=1e-3, sensitivity=1.0)
    tight = GaussianMechanism(epsilon=1.0, delta=1e-8, sensitivity=1.0)
    assert loose.sigma > 0
    assert tight.sigma > loose.sigma


def test_gaussian_rejects_invalid_delta():
    with pytest.raises(ValueError):
        GaussianMechanism(epsilon=1.0, delta=1.5)


def test_clip_state_dict_reduces_norm_when_over_budget():
    state_dict = {"w": torch.ones(10, 10) * 10}
    pre_norm = clip_state_dict_(state_dict, max_norm=1.0)
    assert pre_norm > 1.0

    post_norm = float(torch.sum(state_dict["w"] ** 2)) ** 0.5
    assert post_norm <= 1.0 + 1e-3


def test_clip_state_dict_noop_when_under_budget():
    original = torch.ones(2, 2) * 0.01
    state_dict = {"w": original.clone()}
    clip_state_dict_(state_dict, max_norm=100.0)
    assert torch.allclose(state_dict["w"], original)


def test_clip_parameter_gradients_enforces_max_norm():
    model = nn.Linear(4, 4)
    model(torch.randn(1, 4)).sum().backward()
    for param in model.parameters():
        assert param.grad is not None
        param.grad.mul_(1000)

    clip_parameter_gradients_(model.parameters(), max_norm=1.0)

    total_norm = sum(float((p.grad**2).sum()) for p in model.parameters()) ** 0.5
    assert total_norm <= 1.0 + 1e-2


def test_privacy_accountant_tracks_spend():
    accountant = PrivacyAccountant()
    accountant.spend(round_number=0, epsilon=0.5, delta=1e-6)
    accountant.spend(round_number=1, epsilon=0.3, delta=1e-6)

    assert accountant.spent_epsilon == pytest.approx(0.8)
    assert accountant.remaining_budget(total_epsilon=1.0) == pytest.approx(0.2)
    assert not accountant.is_exhausted(total_epsilon=1.0)
    assert accountant.is_exhausted(total_epsilon=0.8)
