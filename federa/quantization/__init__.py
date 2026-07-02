"""Weight compression: Int8/FP16 quantization, QLoRA hooks, and a unified API."""

from federa.quantization.compression import (
    CompressedPayload,
    compress_state_dict,
    decompress_state_dict,
)
from federa.quantization.fp16 import state_dict_from_fp16, state_dict_to_fp16
from federa.quantization.int8 import (
    Int8Tensor,
    dequantize_tensor_int8,
    quantize_tensor_int8,
)
from federa.quantization.qlora import LoRALinear, inject_lora_adapters, lora_parameters

__all__ = [
    "CompressedPayload",
    "Int8Tensor",
    "LoRALinear",
    "compress_state_dict",
    "decompress_state_dict",
    "dequantize_tensor_int8",
    "inject_lora_adapters",
    "lora_parameters",
    "quantize_tensor_int8",
    "state_dict_from_fp16",
    "state_dict_to_fp16",
]
