import numpy as np
from typing import Tuple
import torch
from bitsandbytes.functional import quantize_q4_0, dequantize_q4_0, QuantState

# Llama.cpp Python code
class Q4_0:
    block_size = 32

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = np.abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -8
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((np.float64(blocks) * np.float64(id)) + np.float64(8.5)).astype(np.uint8).clip(0, 15)

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[:, 0, :] | (qs[:, 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        return (d * qs.astype(np.float32)).reshape(-1)

# Generate random data
np.random.seed(42)
A_np = np.random.randn(100).astype(np.float32)
A_torch = torch.from_numpy(A_np)

# Quantize using llama.cpp code
block_size = Q4_0.block_size
n = A_np.size
n_blocks = (n + block_size - 1) // block_size
pad_size = n_blocks * block_size - n

if pad_size > 0:
    A_np_padded = np.concatenate([A_np, np.zeros(pad_size, dtype=A_np.dtype)])
else:
    A_np_padded = A_np

A_blocks_np = A_np_padded.reshape(n_blocks, block_size)
A_quant_np = Q4_0.quantize_blocks(A_blocks_np)

# Dequantize using llama.cpp code
A_dequant_np = Q4_0.dequantize_blocks(A_quant_np)[:n]

# Quantize using bitsandbytes
A_quant_torch, quant_state = quantize_q4_0(A_torch)

# Dequantize using bitsandbytes
A_dequant_torch = dequantize_q4_0(A_quant_torch, quant_state)

# Convert bitsandbytes outputs to NumPy
A_quant_torch_np = A_quant_torch.numpy()
A_dequant_torch_np = A_dequant_torch.numpy()

# Compare quantized data
quant_equal = np.array_equal(A_quant_np, A_quant_torch_np)
print("Quantized data equal:", quant_equal)

# Compare dequantized data
dequant_diff = np.abs(A_dequant_np - A_dequant_torch_np).mean()
print("Average difference in dequantized data:", dequant_diff)

# Compare original and dequantized data
diff_np = np.abs(A_np - A_dequant_np).mean()
print("Average difference between original and dequantized data (llama.cpp):", diff_np)

diff_torch = np.abs(A_np - A_dequant_torch_np).mean()
print("Average difference between original and dequantized data (bitsandbytes):", diff_torch)