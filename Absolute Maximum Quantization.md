---
aliases:
  - Absmax Quantization
---
References:
- Article: [Introduction to Weight Quantization from Maximme Labonne](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)

Compare with: [[Zero-Point Quantization]]
- ==Zero-point quantization should be slightly better than absmax, but is also more costly to compute==.

With [[Absolute Maximum Quantization|Absmax Quantization]]:
- The original number is divided by the absolute maximum value of the tensor, and then multiplied by a scaling factor (127) to map inputs into the range \[-127, 127\] (for our 8 bit quantization example), before being rounded to the nearest number (we rearrange these terms a bit below).
- If we want to recover the original FP16 values, we divide out INT8 number by the quantization factor, acknowledging some loss of precision due to rounding.
- Example
	- If we had an absolution maximum value of 3.2, a weight of 0.1 would be quantized to ROUND((127/3.2)\*0.1) = ROUND(3.968) = 4
![[Pasted image 20240627120852.png|300]]
```python
def absmax_quantize(X):
	# Calculate scaling factor
	scale = 127/torch.max(torch.abs(X))
	# Quantize
	X_quant = (scale * X).round()
	# Dequantize
	X_dequant = X_quant / scale

	return X_quant.to(torch.int8), X_dequant
```


![[Pasted image 20240627121601.png]]

In this example, we applied quantization techniques to entire layers (per-tensor basis). ==In practice, we often prefer the **vector-wise quantization**, which considers the variability of values in rows and columns inside of the same tensor.==