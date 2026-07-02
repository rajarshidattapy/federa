import torch
import torch.nn as nn

from federa.models.pytorch import wrap_model
from federa.models.serialization import (
    bytes_to_state_dict,
    state_dict_num_parameters,
    state_dict_to_bytes,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_get_set_weights_roundtrip():
    source = wrap_model(_TinyModel())
    target = wrap_model(_TinyModel())

    target.set_weights(source.get_weights())

    for key, value in source.get_weights().items():
        assert torch.equal(value, target.get_weights()[key])


def test_state_dict_bytes_roundtrip():
    weights = wrap_model(_TinyModel()).get_weights()
    restored = bytes_to_state_dict(state_dict_to_bytes(weights))

    for key, value in weights.items():
        assert torch.equal(value, restored[key])


def test_state_dict_num_parameters_counts_weight_and_bias():
    weights = wrap_model(_TinyModel()).get_weights()
    assert state_dict_num_parameters(weights) == 2 * 1 + 1  # 2 weights + 1 bias
