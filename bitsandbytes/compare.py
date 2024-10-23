import numpy as np
import torch
from bitsandbytes.functional import quantize_q4_0, dequantize_q4_0

# Python implementation of Llama.cpp
class Q4_0:
    block_size = 32

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = np.abs(blocks).argmax(axis=-1, keepdims=True)
        max_vals = np.take_along_axis(blocks, imax, axis=-1)

        d = max_vals / -8
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((np.float64(blocks) * np.float64(id)) + np.float64(8.5)).astype(np.uint8).clip(0, 15)

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs_packed = qs[:, 0, :] | (qs[:, 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d.reshape(n_blocks, -1), qs_packed], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d_uint8, qs_packed = np.hsplit(blocks, [2])

        d = d_uint8.view(np.float16).astype(np.float32)

        qs = qs_packed.reshape((n_blocks, -1, 1, cls.block_size // 2))
        shifts = np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = qs >> shifts
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - 8

        return (d * qs.astype(np.float32)).reshape(-1)

# Generate random data
np.random.seed(42)
A_np = np.random.randn(100).astype(np.float32)
A_torch = torch.from_numpy(A_np)

# Define block size
block_size = Q4_0.block_size

# Process A_np to make its length a multiple of block_size
n_elements = A_np.size
n_blocks = n_elements // block_size
remainder = n_elements % block_size

if remainder != 0:
    # Pad the array to make its length a multiple of block_size
    padding_size = block_size - remainder
    A_np_padded = np.pad(A_np, (0, padding_size), mode='constant', constant_values=0)
else:
    A_np_padded = A_np

# Recalculate the number of blocks
n_blocks = A_np_padded.size // block_size

# Reshape A_np_padded into blocks
blocks_np = A_np_padded.reshape(n_blocks, block_size)

# Quantize using Llama.cpp method
quantized_llama = Q4_0.quantize_blocks(blocks_np)

# Quantize using bitsandbytes method
# Need to pad A_torch in the same way
A_torch_padded = torch.from_numpy(A_np_padded)
quantized_bitsandbytes = quantize_q4_0(A_torch_padded)

# Compare quantization results
quantized_bitsandbytes_np = quantized_bitsandbytes.numpy()

if np.array_equal(quantized_bitsandbytes_np, quantized_llama):
    print("Quantization results are identical.")
else:
    print("Quantization results differ.")
    diff_indices = np.where(quantized_bitsandbytes_np != quantized_llama)
    print("Difference indices:", diff_indices)
    print("bitsandbytes quantized values:", quantized_bitsandbytes_np[diff_indices])
    print("Llama.cpp quantized values:", quantized_llama[diff_indices])

# Dequantize using Llama.cpp method
dequantized_llama = Q4_0.dequantize_blocks(quantized_llama)

# Dequantize using bitsandbytes method
dequantized_bitsandbytes = dequantize_q4_0(quantized_bitsandbytes)
dequantized_bitsandbytes_np = dequantized_bitsandbytes.numpy()

# Compare dequantization results
if np.allclose(dequantized_bitsandbytes_np, dequantized_llama, atol=1e-6):
    print("Dequantization results are identical.")
else:
    print("Dequantization results differ.")
    diff_indices = np.where(np.abs(dequantized_bitsandbytes_np - dequantized_llama) > 1e-6)
    print("Difference indices:", diff_indices)
    print("bitsandbytes dequantized values:", dequantized_bitsandbytes_np[diff_indices])
    print("Llama.cpp dequantized values:", dequantized_llama[diff_indices])

# Calculate error compared to original data
original_data = A_np_padded
error_bitsandbytes = np.mean((dequantized_bitsandbytes_np - original_data) ** 2)
error_llama = np.mean((dequantized_llama - original_data) ** 2)
print(f"bitsandbytes dequantization error: {error_bitsandbytes}")
print(f"Llama.cpp dequantization error: {error_llama}")
