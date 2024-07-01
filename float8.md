---
aliases:
  - FP8
---
Reference:
- Notes: [[TinyML (5) - Quantization part 1]]

Most recently, there's a new number that came out to further reduce the precision of NNs, making training faster.
During training we want higher dynamic range, but during inference, we want high precision!
- There's no free lunch: You either get higher dynamic range and less precision, or higher precision and less dynamic range.
![[Pasted image 20240630184554.png]]
(E4M3 = Exponent 4 Bits, Mantissa 3 Bits)
- We use E4M3 for inference/forward pass, because it has smaller dynamic range but more precision.
- WE use E5M2 for backward passes because it has a larger dynamic range (so we can multiply those very large gradients esp. early in training), but less precision.
