import torch
from bitsandbytes.functional import quantize_q4_0, dequantize_q4_0

# Sample tensor
torch.manual_seed(42)
A = torch.randn(100, dtype=torch.float32)

# Quantize
A_quant, quant_state = quantize_q4_0(A)

# Dequantize
A_dequant = dequantize_q4_0(A_quant, quant_state)

# Compare original and dequantized tensors
print("Original Tensor Size:", A.shape)
print("Original Tensor:", A)
print("Dequantized Tensor Size:", A_dequant.shape)
print("Dequantized Tensor:", A_dequant)
print("Difference:", torch.abs(A - A_dequant).mean())
