import torch
import torch.nn as nn

from federa.quantization.compression import compress_state_dict, decompress_state_dict
from federa.quantization.fp16 import state_dict_from_fp16, state_dict_to_fp16
from federa.quantization.int8 import dequantize_tensor_int8, quantize_tensor_int8
from federa.quantization.qlora import (
    LoRALinear,
    dequantize_tensor_int4,
    inject_lora_adapters,
    lora_parameters,
    quantize_tensor_int4,
)


def test_int8_roundtrip_within_tolerance():
    tensor = torch.randn(10, 10)
    quantized = quantize_tensor_int8(tensor)
    restored = dequantize_tensor_int8(quantized)
    assert restored.shape == tensor.shape
    assert torch.allclose(tensor, restored, atol=quantized.scale * 1.5)


def test_int8_zero_tensor_does_not_divide_by_zero():
    tensor = torch.zeros(3, 3)
    restored = dequantize_tensor_int8(quantize_tensor_int8(tensor))
    assert torch.allclose(restored, tensor)


def test_fp16_roundtrip_state_dict_preserves_non_float_dtype():
    state_dict = {"w": torch.randn(4, 4), "count": torch.tensor(5, dtype=torch.int64)}
    half = state_dict_to_fp16(state_dict)
    assert half["w"].dtype == torch.float16

    restored = state_dict_from_fp16(half)
    assert torch.allclose(state_dict["w"], restored["w"], atol=1e-2)
    assert restored["count"].dtype == torch.int64


def test_compression_roundtrip_all_methods():
    state_dict = {"w": torch.randn(5, 5), "b": torch.randn(5)}
    for method in ("none", "fp16", "int8"):
        payload = compress_state_dict(state_dict, method)
        restored = decompress_state_dict(payload)
        for key in state_dict:
            assert torch.allclose(state_dict[key], restored[key], atol=0.1)


def test_int4_roundtrip_within_tolerance():
    tensor = torch.randn(20)
    quantized = quantize_tensor_int4(tensor)
    restored = dequantize_tensor_int4(quantized)
    assert restored.shape == tensor.shape
    assert torch.allclose(tensor, restored, atol=quantized.scale * 1.5)


def test_lora_injection_freezes_base_and_exposes_adapter_params():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    inject_lora_adapters(model, rank=2)

    assert isinstance(model[0], LoRALinear)
    assert isinstance(model[2], LoRALinear)

    adapters = lora_parameters(model)
    assert len(adapters) == 4  # lora_a + lora_b for each of the 2 replaced layers

    for name, param in model.named_parameters():
        if "base" in name:
            assert not param.requires_grad
