References:
- Article: [Introduction to Weight Quantization from Maximme Labonne](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)


With [[Zero-Point Quantization]]:
- We can consider asymmetric input distributions, which is useful when considering (eg) the output of a [[Rectified Linear Unit|ReLU]] function. 
- The input values are first scaled by the total range of values (255) divided by the *difference between the maximum and minimum values*.
	- This distribution is then shifted to map it into the range \[-128, 127\] (notice the extra value, compared to [[Absolute Maximum Quantization|Absmax Quantization]]).
![[Pasted image 20240627120838.png|400]]
Then, we can use these variables to quantize or dequantize our weights:
![[Pasted image 20240627120940.png|400]]
Example
- If we have a maximum value of 3.2 and a minimum value of -3.0, we calculate the *scale* as $255/(3.2+3.0) = 41.13$, and the *zero-point* as $-round(41.13 * -3.0) - 128 = -5$, so our previous weight of 0.1 would be quantized to $round(41.13*0.1-5) = -1$.
- This -1 is very different form the previous value 4 obtained using [[Absolute Maximum Quantization|Absmax Quantization]].
	- ==zero-point quantization should be slightly better than absmax, but is also more costly to compute.==
```python
def zeropoint_quantize(x):
	# Calculate the value range (denominator)
	x_range = torch.max(X) - torch.min(X)
	x_range = 1 if x_range == 0 else x_range

	# Calculate scale
	scale = 255 / x_range

	# Shift by zero-point
	zeropoint = (-scale * torch.min(X) - 128).round()

	# Quantize: scale and round the inputs
	X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

	# Dequantize
	X_dequant = (X_quant - zeropoint) / scale

	return X_quant.to(torch.int8), X_dequant
```


![[Pasted image 20240627121601.png]]

In this example, we applied quantization techniques to entire layers (per-tensor basis). ==In practice, we often prefer the **vector-wise quantization**, which considers the variability of values in rows and columns inside of the same tensor.==